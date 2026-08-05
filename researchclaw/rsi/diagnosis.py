"""Campaign diagnosis and cross-cycle mutation materialization."""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path
from typing import Any

from researchclaw.evolution import LessonEntry
from researchclaw.evolution_aevolve import run_aevolve_cycle
from researchclaw.pipeline._helpers import StageResult
from researchclaw.pipeline.result_validity import assess_experiment_summary

from .evidence import load_lessons, load_stage_results
from .storage import CampaignStore, atomic_write_json, utc_now

_DIAGNOSIS_SYSTEM = """\
You are the campaign-level research director for AutoResearchClaw. Diagnose one
completed pipeline cycle using only the supplied evidence. Recommend precise,
testable changes for the next cycle. Never recommend automatic submission,
publication, or claims unsupported by experiment evidence.

The campaign supervisor already implements autonomous topic selection,
cross-cycle diagnosis, mutation, acceptance, pivoting, and campaign iteration.
The ordinary 23-stage pipeline is responsible for executing the one concrete
selected research topic supplied in selected_topic. Never require experiment
code to reimplement the campaign supervisor or the entire meta-research loop.
When a failure message confuses the campaign meta-brief with the selected
topic, diagnose that as a routing/alignment defect and preserve the concrete
selected hypothesis as the experiment target.

Return ONLY one JSON object with these fields:
- summary: concise diagnosis
- strengths: array of strings
- weaknesses: array of strings
- next_cycle_priorities: array of strings
- prompt_patch: compact guidance to append to every next-cycle stage prompt
- repair_prompt_patch: when the cycle failed, a bounded engineering repair for
  the exact observed failure; otherwise "". It may fix code, dependencies,
  data loading, resource use, or reproducibility, but must never lower gates,
  change safety policy, fabricate evidence, or redefine failure as success
- topic_action: one of "keep", "refine", or "pivot"
- topic_patch: for "refine", a concrete revised topic specification; otherwise ""
- pivot_reason: for "pivot", the evidence-based reason the incumbent topic
  should be rejected; otherwise ""
- preferred_candidate_id: optional existing candidate ID to use for a pivot;
  leave empty to run a fresh >=12-candidate selection
- stop_recommended: boolean
- stop_reason: string

The campaign meta-brief and its no-publication safety boundary are immutable.
Never propose changing them. Default to topic_action="keep". Use "refine"
only for a bounded version of the same falsifiable question, and use "pivot"
only when cycle evidence invalidates novelty, feasibility, or the primary
signal.
"""

_UNSAFE_REPAIR_PATTERNS = (
    r"\b(?:disable|bypass|skip|remove|delete|weaken|lower|relax|ignore|"
    r"override|replace|turn\s+off|no-?op|set)\b.{0,120}"
    r"\b(?:gate|quality[_ -]?gate|threshold|verification|validator|review|citation|safety|"
    r"reviewer|reproducibility|evidence|metric|score|status|export|publish|"
    r"submission)\b",
    r"\b(?:gate|quality[_ -]?gate|threshold|verification|validator|review|reviewer|citation|safety|"
    r"reproducibility|evidence|metric|score|status|export|publish|"
    r"submission)\b.{0,120}\b(?:false|off|zero|no-?op|ignore|override)\b",
    r"\b(?:fabricat|fake|forge|invent|hallucinat|hardcode)\w*\b.{0,80}"
    r"\b(?:result|metric|evidence|citation|output|score)\b",
    r"\b(?:synthetic|default|fallback)\b.{0,80}"
    r"\b(?:metrics?|scores?|results?|evidence)\b",
    r"\b(?:accept|approve|pass)\b.{0,100}"
    r"\b(?:regardless|without|missing|invalid|failed|no evidence)\b",
    r"\b(?:treat|mark|report|declare)\b.{0,80}\b(?:failure|failed|error)\b"
    r".{0,80}\b(?:success|passed|valid)\b",
    r"\b(?:automatic|auto)\w*\b.{0,40}\b(?:publish|submission|release)\b",
    r"\b(?:change|replace|override|rewrite)\b.{0,80}"
    r"\b(?:meta-brief|safety policy|campaign policy)\b",
)


def _safe_repair_prompt(diagnosis: dict[str, Any]) -> tuple[str, str]:
    """Return a bounded repair prompt and a rejection reason.

    Failed-cycle advice is untrusted model output.  Keep this validator
    deliberately conservative: an unsafe or excessively large patch is
    archived in the diagnosis but never injected into a later cycle.
    """

    raw = diagnosis.get("repair_prompt_patch")
    if not str(raw or "").strip():
        # Backward compatibility for directors that still emit the historical
        # prompt_patch field. On failed cycles it is treated as transient,
        # never as accepted permanent campaign guidance.
        raw = diagnosis.get("prompt_patch")
    patch = str(raw or "").strip()
    if not patch:
        return "", "empty"
    if len(patch) > 6000:
        return "", "exceeds_6000_char_limit"
    normalized = " ".join(patch.casefold().split())
    if re.search(
        r"[\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0400-\u04ff]",
        normalized,
    ):
        return "", "non_ascii_control_language_not_allowed"
    for pattern in _UNSAFE_REPAIR_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            return "", f"unsafe_pattern:{pattern}"
    return patch, ""


def apply_failure_repair(
    *,
    store: CampaignStore,
    cycle: int,
    diagnosis: dict[str, Any],
    failure_signature: str,
    recovery_action: str,
    ttl_cycles: int = 2,
) -> dict[str, Any]:
    """Persist a safe, transient engineering repair for a failed cycle."""

    diagnosis_path = store.diagnostics_dir / f"cycle-{cycle:04d}.json"
    atomic_write_json(diagnosis_path, diagnosis)
    patch, rejection = _safe_repair_prompt(diagnosis)
    applied = {
        "repair_applied": False,
        "repair_rejected": bool(rejection and rejection != "empty"),
        "repair_rejection_reason": rejection,
        "diagnosis_path": str(diagnosis_path),
        "failure_signature": failure_signature,
        "recovery_action": recovery_action,
    }
    if not patch or recovery_action != "auto_repair":
        # Never let an older repair leak into a later failure after the
        # director emitted no safe replacement, the validator rejected it, or
        # the supervisor escalated to clean regeneration/quarantine.
        store.shared_repair_patch_path.unlink(missing_ok=True)
        if patch and recovery_action != "auto_repair":
            applied["repair_rejection_reason"] = (
                f"recovery_action:{recovery_action}"
            )
        return applied
    payload = {
        "schema_version": 1,
        "failure_signature": failure_signature,
        "source_cycle": cycle,
        "expires_after_cycle": cycle + 1,
        "recovery_action": recovery_action,
        "repair_prompt_patch": patch,
        "updated_at": utc_now(),
        "automatic_submission_enabled": False,
    }
    atomic_write_json(store.shared_repair_patch_path, payload)
    applied["repair_applied"] = True
    applied["repair_rejection_reason"] = ""
    return applied


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _compact_evidence(evidence: dict[str, Any], brief: str) -> dict[str, Any]:
    experiment_summary = evidence.get("experiment_summary", {})
    if isinstance(experiment_summary, dict):
        validity = assess_experiment_summary(experiment_summary)
        experiment_summary = dict(experiment_summary)
        experiment_summary["result_valid"] = validity.valid
        experiment_summary["validity_reasons"] = validity.reasons
        experiment_summary["valid_conditions"] = sorted(
            condition
            for condition in validity.valid_conditions
            if condition != "__overall__"
        )
        diagnosis = experiment_summary.get("diagnosis")
        if isinstance(diagnosis, dict):
            diagnosis = dict(diagnosis)
            raw_rate = diagnosis.get("completion_rate", 0.0)
            try:
                rate = float(raw_rate)
            except (TypeError, ValueError):
                rate = 0.0
            diagnosis["completion_rate"] = min(1.0, max(0.0, rate))
            experiment_summary["diagnosis"] = diagnosis

    return {
        "campaign_meta_brief": brief[:12000],
        "selected_topic": evidence.get("selected_topic", {}),
        "topic_id": evidence.get("topic_id", ""),
        "pipeline_summary": evidence.get("pipeline_summary", {}),
        "quality_score": evidence.get("quality_score"),
        "quality_report": evidence.get("quality_report", {}),
        "verification_report": evidence.get("verification_report", {}),
        "experiment_summary": experiment_summary,
        "failures": evidence.get("failures", []),
        "lessons": evidence.get("lessons", [])[:30],
        "artifacts": evidence.get("artifacts", []),
    }


def diagnose_cycle(
    *,
    llm: Any,
    evidence: dict[str, Any],
    brief: str,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    """Ask the configured ResearchClaw LLM client for a cycle diagnosis."""

    payload = json.dumps(
        _compact_evidence(evidence, brief),
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    response = _interruptible_call(
        lambda: llm.chat(
            [
                {
                    "role": "user",
                    "content": (
                        "Analyze this campaign cycle and prepare the next "
                        "iteration:\n\n" + payload
                    ),
                }
            ],
            system=_DIAGNOSIS_SYSTEM,
            json_mode=True,
            max_tokens=4000,
            temperature=0.2,
        ),
        cancel_event=cancel_event,
    )
    diagnosis = _json_object(response.content)
    if not diagnosis:
        raise ValueError("campaign diagnosis was not valid JSON")
    return diagnosis


def _interruptible_call(
    call: Any,
    *,
    cancel_event: threading.Event | None,
) -> Any:
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
        name="rsi-interruptible-llm-call",
        daemon=True,
    )
    thread.start()
    while not finished.wait(0.2):
        if cancel_event.is_set():
            raise InterruptedError("campaign control requested during LLM call")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def apply_diagnosis(
    *,
    store: CampaignStore,
    cycle: int,
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    """Persist accepted topic guidance without mutating campaign policy."""

    applied: dict[str, Any] = {
        "brief_updated": False,
        "legacy_brief_patch_ignored": False,
        "topic_patch_updated": False,
        "prompt_updated": False,
        "diagnosis_path": str(store.diagnostics_dir / f"cycle-{cycle:04d}.json"),
    }
    atomic_write_json(Path(applied["diagnosis_path"]), diagnosis)

    brief_patch = str(diagnosis.get("brief_patch", "") or "").strip()
    if brief_patch:
        # Historical payloads may still contain this field. Archive it in the
        # diagnosis, but never overwrite the immutable campaign meta-brief.
        applied["legacy_brief_patch_ignored"] = True

    topic_action = str(
        diagnosis.get("topic_action", "keep") or "keep"
    ).strip().casefold()
    topic_patch = str(diagnosis.get("topic_patch", "") or "").strip()
    if topic_action == "refine" and topic_patch:
        atomic_write_json(
            store.shared_topic_patch_path,
            {
                "schema_version": 1,
                "cycle": cycle,
                "topic_action": "refine",
                "topic_patch": topic_patch,
                "updated_at": utc_now(),
                "automatic_submission_enabled": False,
            },
        )
        applied["topic_patch_updated"] = True

    prompt_patch = str(diagnosis.get("prompt_patch", "") or "").strip()
    if prompt_patch:
        with store.shared_prompt_path.open("a", encoding="utf-8") as handle:
            handle.write(
                f"\n\n## Cycle {cycle} diagnosis ({utc_now()})\n{prompt_patch}\n"
            )
        applied["prompt_updated"] = True
    return applied


def run_campaign_aevolve(
    *,
    llm: Any,
    run_dir: Path,
    skills_dir: Path,
    evidence: dict[str, Any] | None = None,
    cancel_event: threading.Event | None = None,
) -> list[str]:
    """Invoke the existing A-Evolve implementation with reconstructed results."""

    results: list[StageResult] = load_stage_results(run_dir)
    lessons: list[LessonEntry] = load_lessons(run_dir, results)
    quality_context: dict[str, Any] = {}
    if isinstance(evidence, dict):
        scorecard = evidence.get("scorecard")
        components = (
            scorecard.get("components", {})
            if isinstance(scorecard, dict)
            else {}
        )
        weak_components: dict[str, Any] = {}
        if isinstance(components, dict):
            for name, component in components.items():
                if not isinstance(component, dict):
                    continue
                score = component.get("score")
                available = component.get("available")
                if available is False or (
                    isinstance(score, (int, float)) and float(score) < 70.0
                ):
                    weak_components[str(name)] = component
        quality_context = {
            "composite_score": evidence.get("composite_score"),
            "quality_score": evidence.get("quality_score"),
            "comparison": evidence.get("comparison"),
            "hard_failures": (
                scorecard.get("hard_failures", [])
                if isinstance(scorecard, dict)
                else []
            ),
            "weak_components": weak_components,
            "experiment_summary": evidence.get("experiment_summary", {}),
            "verification_report": evidence.get("verification_report", {}),
        }
        if not any(
            value
            for key, value in quality_context.items()
            if key not in {"composite_score", "quality_score"}
        ) and quality_context.get("composite_score") is None:
            quality_context = {}
    return run_aevolve_cycle(
        lessons,
        results,
        llm,
        skills_dir,
        run_dir,
        quality_context=quality_context,
        cancel_event=cancel_event,
    )


def build_llm_client(config_path: Path, role: str = "campaign_director") -> Any:
    """Create the configured campaign-level role client."""

    from researchclaw.config import RCConfig
    from researchclaw.llm.roles import create_role_llm_client

    config = RCConfig.load(config_path, check_paths=False)
    return create_role_llm_client(config, role)
