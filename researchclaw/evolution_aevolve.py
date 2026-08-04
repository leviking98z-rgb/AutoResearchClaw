"""A-Evolve integration: structured observation + gated skill generation.

Implements the Solve → Observe → Evolve → Gate → Reload methodology
from A-Evolve (https://github.com/A-EVO-Lab/a-evolve) as an optional
post-pipeline analysis step.

Differences from the base ``evolution.py`` system:
- Uses LLM for **structured diagnosis** (root cause, frequency, severity)
  rather than keyword-based classification.
- Applies a **quality gate** before accepting generated skills.
- Produces **multiple mutation types**: skills, prompt patches, knowledge
  entries — not just lessons.

Ref: PR #187 by @Gitsamshi
"""

from __future__ import annotations

import json
import logging
import re
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from researchclaw.evolution import LessonEntry
    from researchclaw.llm.client import LLMClient

logger = logging.getLogger(__name__)


def _check_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("campaign control requested during A-Evolve")


def _call_llm(
    call: Callable[[], Any],
    *,
    cancel_event: threading.Event | None,
) -> Any:
    """Run one synchronous LLM request with bounded supervisor cancellation.

    The transport call may still finish in its daemon worker after cancellation,
    but no later A-Evolve phase or mutation write is allowed to run.
    """

    _check_cancelled(cancel_event)
    if cancel_event is None:
        return call()

    result: dict[str, Any] = {}
    finished = threading.Event()

    def worker() -> None:
        try:
            result["value"] = call()
        except BaseException as exc:  # noqa: BLE001
            result["error"] = exc
        finally:
            finished.set()

    thread = threading.Thread(
        target=worker,
        name="aevolve-interruptible-llm-call",
        daemon=True,
    )
    thread.start()
    while not finished.wait(0.2):
        _check_cancelled(cancel_event)
    _check_cancelled(cancel_event)
    if "error" in result:
        raise result["error"]
    return result.get("value")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A structured diagnosis of a failure or quality issue."""

    obs_id: str                # e.g. "OBS-1"
    category: str              # code_bug, timeout, wrong_approach, missing_knowledge, etc.
    root_cause: str
    affected_stages: list[str]
    frequency: int             # how many tasks/stages affected
    severity: str              # blocking, degrading, cosmetic
    description: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Mutation:
    """A proposed change to improve agent performance."""

    mutation_type: str          # "skill", "prompt_patch", "knowledge_entry", "none"
    name: str
    description: str
    content: str
    target_observation: str     # OBS-id this addresses
    gate_passed: bool = False
    gate_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_OBSERVE_SYSTEM = """\
You are a pipeline diagnostician for AutoResearchClaw, a 23-stage autonomous
research pipeline. Analyze the evidence below and produce structured
observations about failures and quality issues.

For each issue, identify:
- category: code_bug | timeout | wrong_approach | missing_knowledge | \
api_misuse | hallucinated_reference | prompt_ambiguity | data_quality | other
- root_cause: What specifically went wrong and why
- affected_stages: Which pipeline stages were affected
- frequency: How many tasks/stages hit this issue
- severity: blocking (pipeline crash) | degrading (wrong result) | \
cosmetic (formatting)

Return a JSON array of observation objects.
"""

_OBSERVE_USER = """\
## Pipeline Run Evidence

### Lesson Log
{lessons_text}

### Stage Results Summary
{stage_summary}

Return ONLY a JSON array of observations. Each object has:
  "obs_id": "OBS-N",
  "category": "...",
  "root_cause": "...",
  "affected_stages": ["..."],
  "frequency": N,
  "severity": "blocking|degrading|cosmetic",
  "description": "..."
"""

_EVOLVE_SYSTEM = """\
You are a skill designer for AutoResearchClaw. Based on diagnostic
observations, propose targeted mutations to prevent recurrence.

Mutation types:
- "skill": A reusable SKILL.md file (for recurring patterns, frequency >= 2)
- "prompt_patch": A short addendum to stage prompts (for ambiguity gaps)
- "knowledge_entry": A factual insight to store (for learned heuristics)
- "none": No action needed (one-off or cosmetic issues)

Rules:
- Target specific failures, not generic advice
- Keep skills under 80 lines, prompt patches under 3 sentences
- Never propose more than 3 mutations per cycle
- Prefix skill names with "arc-aevolve-"
"""

_EVOLVE_USER = """\
## Observations

{observations_text}

## Existing Skills (do not duplicate)
{existing_skills}

Propose mutations. Return ONLY a JSON array. Each object has:
  "mutation_type": "skill|prompt_patch|knowledge_entry|none",
  "name": "arc-aevolve-<slug>",
  "description": "<when to use>",
  "content": "<markdown body or JSON string>",
  "target_observation": "OBS-N"
"""

_GATE_SYSTEM = """\
You are a quality gate for AI agent skill mutations. Evaluate each proposed
mutation and decide whether to accept or reject it.

Check:
1. Specificity: Does it target the observed failure without being overly broad?
2. Testability: Could re-running the failed tasks verify this helps?
3. Blast radius: How much agent behavior does this change? Prefer small.
4. Consistency: Does it contradict existing pipeline behavior?

Return a JSON array with one object per mutation:
  "name": "<mutation name>",
  "passed": true|false,
  "reason": "<explanation>"
"""


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def _format_lessons_for_observe(lessons: list[LessonEntry]) -> str:
    parts: list[str] = []
    for i, lesson in enumerate(lessons, 1):
        parts.append(
            f"{i}. [{lesson.severity}] [{lesson.category}] "
            f"Stage {lesson.stage_name} (#{lesson.stage_num}): "
            f"{lesson.description}"
        )
    return "\n".join(parts) if parts else "(no lessons recorded)"


def _format_stage_summary(results: list[object]) -> str:
    parts: list[str] = []
    for result in results:
        stage_num = int(getattr(result, "stage", 0))
        status = str(getattr(result, "status", "unknown"))
        error = getattr(result, "error", None)
        line = f"Stage {stage_num:02d}: {status}"
        if error:
            line += f" — {str(error)[:120]}"
        parts.append(line)
    return "\n".join(parts) if parts else "(no stages executed)"


def _parse_json_response(text: str) -> list[dict[str, Any]]:
    """Parse LLM JSON response, stripping code fences."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```\w*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            for key in ("observations", "mutations", "results", "data"):
                if key in data and isinstance(data[key], list):
                    return data[key]
        if isinstance(data, list):
            return [x for x in data if isinstance(x, dict)]
    except json.JSONDecodeError:
        logger.warning("A-Evolve: failed to parse LLM response as JSON")
    return []


def observe(
    lessons: list[LessonEntry],
    results: list[object],
    llm: LLMClient,
    quality_context: dict[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[Observation]:
    """Step 2: Diagnose failures with structured LLM analysis."""
    _check_cancelled(cancel_event)
    quality_context = dict(quality_context or {})
    has_quality_signal = bool(quality_context)
    if not lessons and not has_quality_signal and not any(
        "failed" in str(getattr(r, "status", "")).lower() for r in results
    ):
        logger.info("A-Evolve observe: no failures to diagnose")
        return []

    stage_summary = _format_stage_summary(results)
    if quality_context:
        stage_summary += (
            "\n\n### Cycle Quality Evidence\n"
            + json.dumps(
                quality_context,
                ensure_ascii=False,
                indent=2,
                default=str,
            )
        )
    user = _OBSERVE_USER.format(
        lessons_text=_format_lessons_for_observe(lessons),
        stage_summary=stage_summary,
    )
    try:
        resp = _call_llm(
            lambda: llm.chat(
                [{"role": "user", "content": user}],
                system=_OBSERVE_SYSTEM,
                json_mode=True,
                max_tokens=2000,
            ),
            cancel_event=cancel_event,
        )
    except InterruptedError:
        raise
    except Exception:
        logger.warning("A-Evolve observe: LLM call failed", exc_info=True)
        return []

    parsed = _parse_json_response(resp.content)
    observations: list[Observation] = []
    for i, item in enumerate(parsed, 1):
        try:
            observations.append(Observation(
                obs_id=item.get("obs_id", f"OBS-{i}"),
                category=item.get("category", "other"),
                root_cause=item.get("root_cause", ""),
                affected_stages=item.get("affected_stages", []),
                frequency=int(item.get("frequency", 1)),
                severity=item.get("severity", "degrading"),
                description=item.get("description", ""),
            ))
        except (TypeError, ValueError):
            continue
    return observations


def evolve(
    observations: list[Observation],
    llm: LLMClient,
    skills_dir: Path,
    cancel_event: threading.Event | None = None,
) -> list[Mutation]:
    """Step 3: Propose mutations based on observations."""
    _check_cancelled(cancel_event)
    if not observations:
        return []

    obs_text = "\n\n".join(
        f"### {o.obs_id}: [{o.category}] {o.description}\n"
        f"- Root cause: {o.root_cause}\n"
        f"- Stages: {', '.join(o.affected_stages)}\n"
        f"- Frequency: {o.frequency}, Severity: {o.severity}"
        for o in observations
    )
    existing = []
    if skills_dir.is_dir():
        existing = [d.name for d in skills_dir.iterdir() if d.is_dir()]

    user = _EVOLVE_USER.format(
        observations_text=obs_text,
        existing_skills=", ".join(existing[:30]) if existing else "(none)",
    )
    try:
        resp = _call_llm(
            lambda: llm.chat(
                [{"role": "user", "content": user}],
                system=_EVOLVE_SYSTEM,
                json_mode=True,
                max_tokens=3000,
            ),
            cancel_event=cancel_event,
        )
    except InterruptedError:
        raise
    except Exception:
        logger.warning("A-Evolve evolve: LLM call failed", exc_info=True)
        return []

    parsed = _parse_json_response(resp.content)
    mutations: list[Mutation] = []
    for item in parsed[:3]:  # max 3 per cycle
        try:
            mutations.append(Mutation(
                mutation_type=item.get("mutation_type", "none"),
                name=item.get("name", ""),
                description=item.get("description", ""),
                content=item.get("content", ""),
                target_observation=item.get("target_observation", ""),
            ))
        except (TypeError, ValueError):
            continue
    return mutations


def gate(
    mutations: list[Mutation],
    llm: LLMClient,
    cancel_event: threading.Event | None = None,
) -> list[Mutation]:
    """Step 4: Validate mutations before accepting."""
    _check_cancelled(cancel_event)
    if not mutations:
        return []

    # Skip "none" type mutations — they need no gating
    for m in mutations:
        if m.mutation_type == "none":
            m.gate_passed = True
            m.gate_reason = "no-op mutation"
    to_gate = [m for m in mutations if m.mutation_type != "none"]
    if not to_gate:
        return mutations

    gate_input = json.dumps(
        [{"name": m.name, "type": m.mutation_type, "content": m.content[:500]}
         for m in to_gate],
        indent=2,
    )
    try:
        resp = _call_llm(
            lambda: llm.chat(
                [
                    {
                        "role": "user",
                        "content": f"Evaluate these mutations:\n{gate_input}",
                    }
                ],
                system=_GATE_SYSTEM,
                json_mode=True,
                max_tokens=1000,
            ),
            cancel_event=cancel_event,
        )
    except InterruptedError:
        raise
    except Exception:
        logger.warning("A-Evolve gate: LLM call failed, rejecting mutations", exc_info=True)
        for m in mutations:
            if m.mutation_type == "none":
                m.gate_passed = True
                m.gate_reason = "no-op mutation"
            else:
                m.gate_passed = False
                m.gate_reason = "gate failed closed (LLM unavailable)"
        return mutations

    parsed = _parse_json_response(resp.content)
    gate_results = {item.get("name", ""): item for item in parsed}

    for m in mutations:
        if m.mutation_type == "none":
            m.gate_passed = True
            m.gate_reason = "no-op mutation"
            continue
        result = gate_results.get(m.name)
        if not isinstance(result, dict) or "passed" not in result:
            m.gate_passed = False
            m.gate_reason = "gate failed closed (missing or invalid result)"
        else:
            m.gate_passed = bool(result.get("passed", False))
            m.gate_reason = str(result.get("reason", ""))
        if not m.gate_passed:
            logger.info(
                "A-Evolve gate rejected '%s': %s", m.name, m.gate_reason
            )

    return mutations


def reload(
    mutations: list[Mutation],
    observations: list[Observation],
    skills_dir: Path,
    run_dir: Path,
    cancel_event: threading.Event | None = None,
) -> list[str]:
    """Step 5: Apply accepted mutations and record results.

    Returns list of created skill names.
    """
    created: list[str] = []

    for m in mutations:
        _check_cancelled(cancel_event)
        if not m.gate_passed or m.mutation_type == "none":
            continue

        if m.mutation_type == "skill" and m.name and m.content:
            # Write as a SKILL.md in the skills directory
            slug = re.sub(r"[^a-z0-9-]", "-", m.name.lower()).strip("-")
            if not slug:
                continue
            skill_dir = skills_dir / slug
            skill_dir.mkdir(parents=True, exist_ok=True)
            skill_content = (
                f"---\nname: {slug}\n"
                f"description: {m.description}\n"
                f"metadata:\n"
                f"  category: a-evolve\n"
                f"  source_observation: {m.target_observation}\n"
                f"  trigger-keywords: \"research,pipeline,quality,experiment,"
                f"{m.target_observation.lower()}\"\n"
                f"  applicable-stages: \"1,2,3,4,5,6,7,8,9,10,11,12,"
                f"13,14,15,16,17,18,19,20,21,22,23\"\n"
                f"  priority: \"2\"\n"
                f"---\n\n{m.content}\n"
            )
            (skill_dir / "SKILL.md").write_text(skill_content, encoding="utf-8")
            created.append(slug)
            logger.info("A-Evolve: created skill '%s'", slug)

        elif m.mutation_type == "prompt_patch" and m.content:
            # Append to a prompt patches file in the run's evolution dir
            patches_file = run_dir / "evolution" / "prompt_patches.md"
            patches_file.parent.mkdir(parents=True, exist_ok=True)
            with patches_file.open("a", encoding="utf-8") as f:
                f.write(f"\n## {m.name}\n{m.content}\n")
            logger.info("A-Evolve: wrote prompt patch '%s'", m.name)

        elif m.mutation_type == "knowledge_entry" and m.content:
            # Append to knowledge entries JSONL
            kb_file = run_dir / "evolution" / "knowledge_entries.jsonl"
            kb_file.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "name": m.name,
                "content": m.content,
                "source_observation": m.target_observation,
                "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
            }
            with kb_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            logger.info("A-Evolve: wrote knowledge entry '%s'", m.name)

    # Write the evolution log
    _check_cancelled(cancel_event)
    log_file = run_dir / "evolution" / "aevolve_log.json"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_data = {
        "timestamp": datetime.now(UTC).isoformat(timespec="seconds"),
        "observations": [o.to_dict() for o in observations],
        "mutations": [m.to_dict() for m in mutations],
        "skills_created": created,
    }
    log_file.write_text(json.dumps(log_data, indent=2), encoding="utf-8")

    return created


# ---------------------------------------------------------------------------
# Main entry point — runs the full Solve → Observe → Evolve → Gate → Reload
# ---------------------------------------------------------------------------

def run_aevolve_cycle(
    lessons: list[LessonEntry],
    results: list[object],
    llm: LLMClient,
    skills_dir: Path,
    run_dir: Path,
    quality_context: dict[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[str]:
    """Execute a full A-Evolve cycle after a pipeline run.

    Args:
        lessons: Extracted lessons from ``evolution.extract_lessons()``.
        results: List of ``StageResult`` objects from the pipeline run.
        llm: LLM client for diagnosis and skill generation.
        skills_dir: Directory to write generated skills.
        run_dir: Pipeline run directory for artifact storage.

    Returns:
        List of created skill names (empty if nothing was generated).
    """
    # Step 1: Solve — evidence is already collected (lessons + results)
    logger.info("A-Evolve: starting cycle (%d lessons, %d stages)", len(lessons), len(results))

    # Step 2: Observe
    observations = observe(
        lessons,
        results,
        llm,
        quality_context=quality_context,
        cancel_event=cancel_event,
    )
    if not observations:
        logger.info("A-Evolve: no observations — skipping evolve cycle")
        return []
    logger.info("A-Evolve: %d observations produced", len(observations))

    # Step 3: Evolve
    mutations = evolve(
        observations,
        llm,
        skills_dir,
        cancel_event=cancel_event,
    )
    if not mutations:
        logger.info("A-Evolve: no mutations proposed")
        # Still write log with observations
        reload(
            [],
            observations,
            skills_dir,
            run_dir,
            cancel_event=cancel_event,
        )
        return []
    logger.info("A-Evolve: %d mutations proposed", len(mutations))

    # Step 4: Gate
    mutations = gate(mutations, llm, cancel_event=cancel_event)
    accepted = [m for m in mutations if m.gate_passed and m.mutation_type != "none"]
    logger.info(
        "A-Evolve: %d/%d mutations passed gate",
        len(accepted), len(mutations),
    )

    # Step 5: Reload
    created = reload(
        mutations,
        observations,
        skills_dir,
        run_dir,
        cancel_event=cancel_event,
    )
    if created:
        logger.info("A-Evolve: created %d skills: %s", len(created), created)

    return created
