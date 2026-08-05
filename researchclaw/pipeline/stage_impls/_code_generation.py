"""Stage 10: Code generation."""

from __future__ import annotations

import ast
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.experiment.validator import (
    format_issues_for_llm,
    validate_code,
)
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _bind_stage_role,
    _chat_with_prompt,
    _extract_code_block,
    _extract_multi_file_blocks,
    _get_evolution_overlay,
    _load_hardware_profile,
    _read_prior_artifact,
    _safe_json_loads,
    _utcnow_iso,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)

_GENERATED_RUNTIME_MODULES = {
    # Injected by every supported experiment sandbox before execution.
    "experiment_harness",
}


def _missing_generated_imports(files: dict[str, str]) -> list[tuple[str, str]]:
    """Return imports that are neither generated nor available at runtime."""

    known_modules = {
        filename.removesuffix(".py")
        for filename in files
        if filename.endswith(".py")
    }
    common_third_party = {
        "numpy", "np", "torch", "torchvision", "gymnasium", "gym",
        "sklearn", "scipy", "pandas", "matplotlib", "PIL", "tqdm",
        "einops", "timm", "transformers", "datasets", "peft",
        "stable_baselines3",
    }
    available_modules = (
        set(sys.stdlib_module_names)
        | common_third_party
        | _GENERATED_RUNTIME_MODULES
    )
    missing: list[tuple[str, str]] = []
    for filename, code in files.items():
        if not filename.endswith(".py"):
            continue
        for module in re.findall(
            r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
            code,
            re.MULTILINE,
        ):
            if (
                module not in known_modules
                and module not in available_modules
                and not module.startswith("_")
            ):
                missing.append((filename, module))
    return missing


def _confirmed_ablation_duplicates(review: Any) -> bool:
    """Require an explicit boolean and concrete duplicate-pair evidence."""

    if not isinstance(review, dict) or review.get("has_duplicates") is not True:
        return False
    pairs = review.get("duplicate_pairs")
    if not isinstance(pairs, list) or not pairs:
        return False
    return any(
        isinstance(pair, dict)
        and str(pair.get("left", "")).strip()
        and str(pair.get("right", "")).strip()
        and str(pair.get("evidence", "")).strip()
        for pair in pairs
    )


def _ablation_review_context(files: dict[str, str]) -> str:
    """Show the reviewer orchestration and implementation files, not a prefix."""

    ordered_names = sorted(
        files,
        key=lambda name: (
            0 if name == "main.py" else
            1 if "model" in name.casefold() else
            2,
            name,
        ),
    )
    sections: list[str] = []
    remaining = 48_000
    for name in ordered_names:
        if remaining <= 0:
            break
        code = files[name]
        excerpt = code[:remaining]
        sections.append(f"# --- {name} ---\n{excerpt}")
        remaining -= len(excerpt)
    return "\n\n".join(sections)

# Improvement G: Continuous-action environments that are incompatible with DQN
_CONTINUOUS_ENVS = {
    "pendulum", "halfcheetah", "hopper", "walker2d", "ant", "humanoid",
    "swimmer", "reacher", "invertedpendulum", "inverteddoublependulum",
    "mountaincarcontinuous", "lunarlander-continuous",
}

_GENERIC_MODEL_MARKERS = (
    "language model",
    "llm",
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "phi-",
    "transformers",
    "automodelfor",
    "autotokenizer",
    "vllm",
)
_LLM_SUBJECT_MARKERS = (
    "language model",
    "llm",
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "phi-",
    "transformers",
    "vllm",
)
_GENERIC_DATASET_MARKERS = (
    "dataset",
    "benchmark",
    "gsm8k",
    "math",
    "mbpp",
    "humaneval",
    "mmlu",
    "arc",
    "hellaswag",
    "cifar",
    "mnist",
    "imagenet",
    "load_dataset",
    "torchvision.datasets",
)
_NAMED_BENCHMARK_MARKERS = (
    "gsm8k",
    "math",
    "mbpp",
    "humaneval",
    "mmlu",
    "arc",
    "hellaswag",
    "cifar",
    "mnist",
    "imagenet",
)
_MODEL_EXECUTION_MARKERS = (
    "from_pretrained(",
    "automodelfor",
    "autotokenizer",
    "pipeline(",
    "generate(",
    "vllm",
    "litellm",
    "openai(",
    "anthropic(",
)
_DATASET_EXECUTION_MARKERS = (
    "load_dataset(",
    "torchvision.datasets.",
    "datasets.",
    "read_csv(",
    "read_json(",
    "json.load(",
)
_LLM_PROXY_MARKERS = (
    "fashionmnist",
    "fashion mnist",
    "mnist",
    "cifar",
    "imagenet",
    "small mlp",
    "multilayer perceptron",
    "image classification",
    "torchvision.datasets",
)
_LLM_MODEL_IMPLEMENTATION_MARKERS = (
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "phi-",
    "from_pretrained(",
    "automodelforcausallm",
    "vllm",
    "litellm",
    "openai(",
    "anthropic(",
)
_LLM_BENCHMARK_IMPLEMENTATION_MARKERS = (
    "gsm8k",
    "math",
    "mbpp",
    "humaneval",
)
_EXPLICIT_SIMULATION_PATTERNS = (
    (
        "synthetic_data_generator",
        re.compile(
            r"\b(?:generate|generated|generating|use|using|with)\b"
            r".{0,80}\bsynthetic\b.{0,40}"
            r"\b(?:data|dataset|trajectory|trace|benchmark|example|sample|dgp)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "simulated_research_subject",
        re.compile(
            r"\b(?:simulat(?:e|ed|ing|ion)|mock(?:ed|ing)?)\b"
            r".{0,80}\b(?:llm|language model|model inference|self[- ]?"
            r"(?:refinement|improvement|training)|trajectory|trace|benchmark|"
            r"dataset|correctness|"
            r"calibration|accuracy|reward)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "synthetic_research_subject",
        re.compile(
            r"\bsynthetic\b.{0,100}\b(?:llm|language model|model|self[- ]?"
            r"(?:refinement|improvement|training)|trajectory|trace|benchmark|"
            r"dataset|data|dgp)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "synthetic_benchmark_generator",
        re.compile(
            r"\bmake_(?:moons|classification|blobs|regression|circles)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "substituted_production_experiment",
        re.compile(
            r"\b(?:in|for)\s+(?:a\s+)?(?:real|full|actual|production)\s+"
            r"experiment\b.{0,160}\b(?:replace|replaced|swap|use|load|run)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "placeholder_or_proxy_experiment",
        re.compile(
            r"\b(?:placeholder|toy|proxy)\b.{0,60}"
            r"\b(?:experiment|metric|dataset|trajectory|model|benchmark)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
    (
        "random_scientific_outcomes",
        re.compile(
            r"\b(?:rng|random|np\.random|torch\.rand)\b.{0,180}"
            r"\b(?:accuracy|correctness|calibration|ece|reward|score|"
            r"regression|collapse|confidence|prediction)\b",
            re.IGNORECASE | re.DOTALL,
        ),
    ),
)


def _normalize_contract_values(value: Any) -> list[str]:
    """Extract searchable contract values without assuming one schema."""

    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, dict):
        values: list[str] = []
        for key, item in value.items():
            if str(key).strip():
                values.append(str(key))
            values.extend(_normalize_contract_values(item))
        return values
    if isinstance(value, (list, tuple, set)):
        values = []
        for item in value:
            values.extend(_normalize_contract_values(item))
        return values
    return [str(value)]


def _contract_markers(values: Any) -> list[str]:
    """Return distinctive declared model/dataset markers for code search."""

    markers: list[str] = []
    seen: set[str] = set()
    ignored = {
        "development",
        "held-out",
        "held out",
        "family",
        "open",
        "checkpoint",
        "where",
        "license",
        "permits",
        "class",
        "model",
        "dataset",
        "benchmark",
    }
    for raw in _normalize_contract_values(values):
        lowered = raw.casefold()
        candidates = [lowered]
        candidates.extend(
            token
            for token in re.findall(r"[a-z][a-z0-9.+_-]{2,}", lowered)
            if token not in ignored and len(token) >= 4
        )
        for candidate in candidates:
            normalized = candidate.strip(" /,;:()[]{}")
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            markers.append(normalized)
    return markers


def _load_scientific_contract(
    run_dir: Path,
    config: RCConfig,
) -> dict[str, Any]:
    """Load the authoritative selected topic, with Stage-9 as fallback."""

    selected_text = ""
    for candidate in (
        run_dir / "selected_topic.json",
        run_dir / "stage-01" / "selected_topic.json",
    ):
        if candidate.is_file():
            selected_text = candidate.read_text(encoding="utf-8")
            break
    if not selected_text:
        selected_text = (
            _read_prior_artifact(run_dir, "selected_topic.json") or "{}"
        )
    selected = _safe_json_loads(selected_text, {})
    if not isinstance(selected, dict):
        selected = {}
    try:
        import yaml

        plan = yaml.safe_load(
            _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
        )
    except Exception:  # noqa: BLE001
        plan = None
    if not isinstance(plan, dict):
        plan = {}
    embedded = plan.get("selected_topic_contract")
    if isinstance(embedded, dict):
        for key, value in embedded.items():
            selected.setdefault(key, value)
    for key in (
        "title",
        "research_question",
        "falsifiable_hypothesis",
        "datasets",
        "models",
        "primary_metric",
        "cheap_pilot",
    ):
        if selected.get(key) in (None, "", [], {}):
            value = plan.get(key)
            if value not in (None, "", [], {}):
                selected[key] = value
    selected.setdefault("title", config.research.topic)
    return selected


def _implementation_contract(
    selected_topic: dict[str, Any],
    fallback_topic: str,
) -> dict[str, Any]:
    """Compile the selected topic into immutable Stage-10 requirements."""

    contract_text = " ".join(
        _normalize_contract_values(selected_topic)
    ).casefold()
    envelope = selected_topic.get("pilot_envelope")
    if not isinstance(envelope, dict):
        envelope = {}

    def positive_int(key: str) -> int | None:
        try:
            value = int(envelope.get(key))
        except (TypeError, ValueError):
            return None
        return value if value > 0 else None

    return {
        "schema_version": 1,
        "topic_id": str(selected_topic.get("id", "") or ""),
        "title": str(
            selected_topic.get("title", fallback_topic) or fallback_topic
        ),
        "research_question": str(
            selected_topic.get("research_question", "") or ""
        ),
        "falsifiable_hypothesis": str(
            selected_topic.get("falsifiable_hypothesis", "") or ""
        ),
        "models": list(
            selected_topic.get("models")
            if isinstance(selected_topic.get("models"), list)
            else _normalize_contract_values(selected_topic.get("models"))
        ),
        "datasets": list(
            selected_topic.get("datasets")
            if isinstance(selected_topic.get("datasets"), list)
            else _normalize_contract_values(selected_topic.get("datasets"))
        ),
        "primary_metric": str(
            selected_topic.get("primary_metric", "") or ""
        ),
        "cheap_pilot": str(selected_topic.get("cheap_pilot", "") or ""),
        "required_capabilities": {
            "real_model_execution": any(
                marker in contract_text for marker in _GENERIC_MODEL_MARKERS
            ),
            "real_dataset_execution": any(
                marker in contract_text for marker in _GENERIC_DATASET_MARKERS
            ),
            "llm_subject": any(
                marker in contract_text for marker in _LLM_SUBJECT_MARKERS
            ),
            "named_llm_benchmark": any(
                marker in contract_text
                for marker in _NAMED_BENCHMARK_MARKERS
            ),
            "calibration_driven_acceptance_gate": any(
                marker in contract_text
                for marker in (
                    "acceptance gate",
                    "acceptance/stopping gate",
                    "calibration-aware gate",
                    "rollback gate",
                )
            ),
        },
        "pilot_envelope": {
            "max_gpus": positive_int("max_gpus"),
            "max_seeds": positive_int("max_seeds"),
            "max_iterations": positive_int("max_iterations"),
            "max_examples": positive_int("max_examples"),
        },
        "forbidden_substitutions": [
            "synthetic or simulated scientific outcomes",
            "toy, proxy, image-classification, or small-MLP replacement",
            "mocked benchmark scores or hard-coded metrics",
            "post-hoc best-round selection presented as an online gate",
        ],
    }


def _assess_implementation_manifest(
    manifest: dict[str, Any],
    contract: dict[str, Any],
) -> dict[str, Any]:
    """Deterministically reject a design manifest that violates the contract."""

    implementation = manifest.get("implementation")
    if not isinstance(implementation, dict):
        implementation = manifest
    serialized = json.dumps(
        implementation,
        ensure_ascii=False,
        sort_keys=True,
    ).casefold()
    capabilities = contract.get("required_capabilities")
    if not isinstance(capabilities, dict):
        capabilities = {}
    envelope = contract.get("pilot_envelope")
    if not isinstance(envelope, dict):
        envelope = {}

    def as_positive_int(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def first_positive(*keys: str) -> int | None:
        for key in keys:
            value = as_positive_int(implementation.get(key))
            if value is not None:
                return value
        resources = implementation.get("resources")
        if isinstance(resources, dict):
            for key in keys:
                value = as_positive_int(resources.get(key))
                if value is not None:
                    return value
        return None

    model = str(implementation.get("model", "") or "").strip()
    benchmark = str(
        implementation.get("benchmark", "")
        or implementation.get("dataset", "")
        or ""
    ).strip()
    calibration_metric = str(
        implementation.get("calibration_metric", "") or ""
    ).strip()
    acceptance_rule = str(
        implementation.get("acceptance_rule", "")
        or implementation.get("gate_rule", "")
        or ""
    ).strip()
    rollback_behavior = str(
        implementation.get("rollback_behavior", "")
        or implementation.get("rejection_behavior", "")
        or ""
    ).strip()
    gpus = first_positive("gpus", "gpu_count", "max_gpus")
    seeds = first_positive("seeds", "seed_count", "max_seeds")
    iterations = first_positive(
        "iterations",
        "iteration_count",
        "max_iterations",
    )
    examples = first_positive(
        "examples",
        "example_count",
        "max_examples",
    )

    reasons: list[str] = []
    if capabilities.get("real_model_execution") and not model:
        reasons.append("manifest omits the executable model/checkpoint")
    if capabilities.get("real_dataset_execution") and not benchmark:
        reasons.append("manifest omits the executable benchmark/dataset")
    if capabilities.get("llm_subject") and not any(
        marker in serialized for marker in _LLM_MODEL_IMPLEMENTATION_MARKERS
    ):
        reasons.append(
            "manifest does not name a concrete LLM checkpoint or inference adapter"
        )
    if capabilities.get("named_llm_benchmark") and not any(
        marker in serialized
        for marker in _LLM_BENCHMARK_IMPLEMENTATION_MARKERS
    ):
        reasons.append(
            "manifest does not select GSM8K/MATH/MBPP/HumanEval"
        )
    if capabilities.get("calibration_driven_acceptance_gate"):
        if not calibration_metric:
            reasons.append("manifest omits the calibration metric")
        if not acceptance_rule:
            reasons.append("manifest omits the online accept/reject rule")
        if not rollback_behavior:
            reasons.append("manifest omits rejection/rollback behavior")
    if any(
        marker in serialized
        for marker in (
            "synthetic",
            "simulate",
            "simulation",
            "toy model",
            "small mlp",
            "fashionmnist",
            "mnist",
            "cifar",
            "imagenet",
            "mock",
            "proxy benchmark",
        )
    ):
        reasons.append("manifest proposes a forbidden proxy or simulation")

    observed = {
        "model": model,
        "benchmark": benchmark,
        "calibration_metric": calibration_metric,
        "acceptance_rule": acceptance_rule,
        "rollback_behavior": rollback_behavior,
        "gpus": gpus,
        "seeds": seeds,
        "iterations": iterations,
        "examples": examples,
    }
    for observed_key, limit_key in (
        ("gpus", "max_gpus"),
        ("seeds", "max_seeds"),
        ("iterations", "max_iterations"),
        ("examples", "max_examples"),
    ):
        limit = as_positive_int(envelope.get(limit_key))
        value = observed[observed_key]
        if limit is not None and value is None:
            reasons.append(
                f"manifest omits bounded {observed_key} required by pilot envelope"
            )
        elif limit is not None and value is not None and value > limit:
            reasons.append(
                f"manifest exceeds pilot {observed_key} ({value} > {limit})"
            )

    return {
        "approved": not reasons,
        "reasons": reasons,
        "observed": observed,
        "contract_topic_id": contract.get("topic_id", ""),
        "checked_at": _utcnow_iso(),
    }


def _implementation_manifest_prompt(
    contract: dict[str, Any],
    exp_plan: str,
    previous_report: dict[str, Any] | None = None,
) -> str:
    """Create a compact design-before-code prompt for the decision model."""

    retry_context = ""
    if previous_report:
        retry_context = (
            "\n\nPREVIOUS MANIFEST REJECTION:\n"
            + json.dumps(previous_report, ensure_ascii=False, indent=2)
        )
    return (
        "Before any experiment code is generated, produce a minimal "
        "implementation manifest that obeys the authoritative scientific "
        "contract exactly. Do not simplify the scientific subject into a "
        "proxy or simulation.\n\n"
        "AUTHORITATIVE CONTRACT:\n"
        f"{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        "STAGE-9 EXPERIMENT PLAN:\n"
        f"{exp_plan[:12000]}\n"
        f"{retry_context}\n\n"
        "Return JSON only with this exact top-level structure:\n"
        "{\n"
        '  "implementation": {\n'
        '    "model": "exact checkpoint or API adapter",\n'
        '    "benchmark": "exact benchmark and split",\n'
        '    "examples": 1,\n'
        '    "seeds": 1,\n'
        '    "iterations": 1,\n'
        '    "gpus": 1,\n'
        '    "calibration_metric": "exact computable metric",\n'
        '    "acceptance_rule": "online rule evaluated before accepting update",\n'
        '    "rollback_behavior": "what happens when the update is rejected",\n'
        '    "raw_artifacts": ["prompts", "outputs", "scores", "token counts"],\n'
        '    "files": ["main.py"]\n'
        "  },\n"
        '  "rationale": "brief explanation"\n'
        "}\n"
        "All resource counts must be explicit integers no greater than the "
        "pilot envelope. If the contract requires a real LLM and benchmark, "
        "name them explicitly."
    )


def _prepare_implementation_manifest(
    stage_dir: Path,
    contract: dict[str, Any],
    exp_plan: str,
    llm: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load or create a decision-approved implementation manifest."""

    manifest_path = stage_dir / "implementation_manifest.json"
    report_path = stage_dir / "implementation_manifest_gate.json"
    if manifest_path.exists():
        cached = _safe_json_loads(
            manifest_path.read_text(encoding="utf-8"),
            {},
        )
        if isinstance(cached, dict):
            report = _assess_implementation_manifest(cached, contract)
            report["source"] = "cached"
            report_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            if report["approved"]:
                return cached, report

    decision_llm = llm
    for_role = getattr(llm, "for_role", None)
    if callable(for_role):
        decision_llm = for_role(
            "implementation_auditor",
            stage=Stage.CODE_GENERATION,
        )

    last_report: dict[str, Any] | None = None
    last_manifest: dict[str, Any] = {}
    for attempt in range(2):
        response = decision_llm.chat(
            [{
                "role": "user",
                "content": _implementation_manifest_prompt(
                    contract,
                    exp_plan,
                    previous_report=last_report,
                ),
            }],
            system=(
                "You are the decision authority for a scientific code "
                "generation gate. Produce a precise implementation manifest; "
                "never replace a real experiment with a proxy."
            ),
            max_tokens=2048,
            json_mode=True,
        )
        parsed = _safe_json_loads(response.content, {})
        last_manifest = parsed if isinstance(parsed, dict) else {}
        last_report = _assess_implementation_manifest(
            last_manifest,
            contract,
        )
        last_report["source"] = "generated"
        last_report["attempt"] = attempt + 1
        manifest_path.write_text(
            json.dumps(last_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(last_report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if last_report["approved"]:
            break
    return last_manifest, last_report or {
        "approved": False,
        "reasons": ["manifest generation failed"],
    }


def _implementation_guidance(
    contract: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Render immutable contract and approved manifest for every code path."""

    return (
        "\n\n## AUTHORITATIVE SCIENTIFIC CONTRACT — NON-NEGOTIABLE\n"
        f"{json.dumps(contract, ensure_ascii=False, indent=2)}\n\n"
        "## DECISION-APPROVED IMPLEMENTATION MANIFEST — IMPLEMENT EXACTLY\n"
        f"{json.dumps(manifest, ensure_ascii=False, indent=2)}\n\n"
        "Do not change the selected model, benchmark, calibration metric, "
        "accept/reject rule, rollback behavior, or pilot resource counts. "
        "Do not substitute synthetic data, a toy model, or a proxy task."
    )


def _assess_scientific_code_alignment(
    files: dict[str, str],
    selected_topic: dict[str, Any],
    fallback_topic: str,
) -> dict[str, Any]:
    """Detect code that replaces the declared experiment with a simulation.

    Random seeds and Monte Carlo procedures are not rejected by themselves.
    The fail-closed decision requires explicit replacement/simulation language
    or random generation of scientific outcomes, plus missing evidence that the
    declared model/dataset is actually loaded or executed.
    """

    code_files = {
        name: code
        for name, code in files.items()
        if name.endswith(".py")
    }
    combined = "\n\n".join(
        f"# --- {name} ---\n{code}"
        for name, code in sorted(code_files.items())
    )
    lowered = combined.casefold()
    authoritative = bool(str(selected_topic.get("id", "") or "").strip())
    declared_models = _contract_markers(selected_topic.get("models"))
    declared_datasets = _contract_markers(selected_topic.get("datasets"))
    contract_text = " ".join(
        _normalize_contract_values(
            {
                "title": selected_topic.get("title", fallback_topic),
                "research_question": selected_topic.get("research_question"),
                "hypothesis": selected_topic.get("falsifiable_hypothesis"),
                "models": selected_topic.get("models"),
                "datasets": selected_topic.get("datasets"),
                "cheap_pilot": selected_topic.get("cheap_pilot"),
            }
        )
    ).casefold()
    requires_model = bool(
        declared_models
        or any(marker in contract_text for marker in _GENERIC_MODEL_MARKERS)
    )
    requires_dataset = bool(
        declared_datasets
        or any(marker in contract_text for marker in _GENERIC_DATASET_MARKERS)
    )
    requires_llm_subject = any(
        marker in contract_text for marker in _LLM_SUBJECT_MARKERS
    )
    requires_named_benchmark = any(
        marker in contract_text for marker in _NAMED_BENCHMARK_MARKERS
    )
    requires_acceptance_gate = bool(
        requires_llm_subject
        and any(
            marker in contract_text
            for marker in (
                "acceptance gate",
                "acceptance/stopping gate",
                "calibration-aware gate",
                "rollback gate",
            )
        )
    )

    model_markers_found = sorted(
        marker for marker in declared_models if marker in lowered
    )
    dataset_markers_found = sorted(
        marker for marker in declared_datasets if marker in lowered
    )
    model_execution_markers = sorted(
        marker for marker in _MODEL_EXECUTION_MARKERS if marker in lowered
    )
    dataset_execution_markers = sorted(
        marker for marker in _DATASET_EXECUTION_MARKERS if marker in lowered
    )
    proxy_markers_found = sorted(
        marker for marker in _LLM_PROXY_MARKERS if marker in lowered
    )
    llm_model_implementation_markers = sorted(
        marker
        for marker in _LLM_MODEL_IMPLEMENTATION_MARKERS
        if marker in lowered
    )
    llm_benchmark_implementation_markers = sorted(
        marker
        for marker in _LLM_BENCHMARK_IMPLEMENTATION_MARKERS
        if marker in lowered
    )
    acceptance_gate_markers = sorted(
        marker
        for marker in (
            "acceptance_gate",
            "acceptancegate",
            "calibrationawaregate",
            "accept_iter",
            "accept, reason",
            "should_accept",
            "accept_update",
            "reject_update",
            "accepted:",
            "rejected:",
            "retain iteration",
            "rollback",
            "stopping_gate",
        )
        if marker in lowered
    )
    calibration_markers = sorted(
        marker
        for marker in (
            "expected_calibration_error",
            "calibration_error",
            "calibration_gap",
            "brier_score",
            "brier0",
            "brier1",
            "brier_delta",
            "reliability_diagram",
            "ece",
        )
        if marker in lowered
    )
    generic_model_subject_found = any(
        marker in lowered for marker in _GENERIC_MODEL_MARKERS
    )
    generic_dataset_subject_found = any(
        marker in lowered for marker in _GENERIC_DATASET_MARKERS
    )
    simulation_hits = [
        {
            "kind": kind,
            "match": " ".join(match.group(0).split())[:240],
        }
        for kind, pattern in _EXPLICIT_SIMULATION_PATTERNS
        if (match := pattern.search(combined)) is not None
    ]
    fallback_source = bool(
        "fallback experiment: parameter sweep on a synthetic objective"
        in lowered
    )
    missing_model_execution = bool(
        authoritative
        and requires_model
        and not (
            model_execution_markers
            and (model_markers_found or generic_model_subject_found)
        )
    )
    missing_dataset_execution = bool(
        authoritative
        and requires_dataset
        and not (
            dataset_execution_markers
            and (dataset_markers_found or generic_dataset_subject_found)
        )
    )

    reasons: list[str] = []
    if fallback_source:
        reasons.append(
            "generic synthetic fallback code was generated instead of the "
            "authoritative experiment"
        )
    if authoritative and simulation_hits and (
        missing_model_execution or missing_dataset_execution
    ):
        reasons.append(
            "generated code explicitly simulates or proxies the scientific "
            "subject while omitting executable paths for declared real "
            "models or datasets"
        )
    if authoritative and requires_llm_subject and missing_model_execution:
        reasons.append(
            "authoritative LLM experiment declares a real checkpoint/API but "
            "generated code has no executable model loading or inference path"
        )
    if (
        authoritative
        and requires_named_benchmark
        and missing_dataset_execution
    ):
        reasons.append(
            "authoritative experiment declares a named real benchmark but "
            "generated code has no executable benchmark loading path"
        )
    if authoritative and requires_llm_subject and proxy_markers_found:
        reasons.append(
            "authoritative LLM experiment was replaced with an image/"
            "small-model proxy: " + ", ".join(proxy_markers_found)
        )
    if (
        authoritative
        and requires_llm_subject
        and not llm_model_implementation_markers
    ):
        reasons.append(
            "authoritative LLM experiment has no concrete LLM checkpoint, "
            "causal-LM loader, or executable LLM inference adapter"
        )
    if (
        authoritative
        and requires_named_benchmark
        and any(
            marker in contract_text
            for marker in _LLM_BENCHMARK_IMPLEMENTATION_MARKERS
        )
        and not llm_benchmark_implementation_markers
    ):
        reasons.append(
            "generated code omits every declared LLM benchmark family "
            "(GSM8K/MATH/MBPP/HumanEval)"
        )
    if (
        authoritative
        and requires_acceptance_gate
        and (not acceptance_gate_markers or not calibration_markers)
    ):
        reasons.append(
            "authoritative calibration-aware RSI topic has no executable "
            "accept/reject or rollback gate driven by a calibration metric"
        )

    return {
        "aligned": not reasons,
        "authoritative_contract": authoritative,
        "selected_topic_id": str(selected_topic.get("id", "") or ""),
        "selected_topic_title": str(
            selected_topic.get("title", fallback_topic) or fallback_topic
        ),
        "requires_real_model": requires_model,
        "requires_real_dataset": requires_dataset,
        "requires_llm_subject": requires_llm_subject,
        "requires_named_benchmark": requires_named_benchmark,
        "requires_acceptance_gate": requires_acceptance_gate,
        "declared_model_markers": declared_models,
        "declared_dataset_markers": declared_datasets,
        "model_markers_found": model_markers_found,
        "dataset_markers_found": dataset_markers_found,
        "model_execution_markers": model_execution_markers,
        "dataset_execution_markers": dataset_execution_markers,
        "proxy_markers_found": proxy_markers_found,
        "llm_model_implementation_markers": (
            llm_model_implementation_markers
        ),
        "llm_benchmark_implementation_markers": (
            llm_benchmark_implementation_markers
        ),
        "acceptance_gate_markers": acceptance_gate_markers,
        "calibration_markers": calibration_markers,
        "generic_model_subject_found": generic_model_subject_found,
        "generic_dataset_subject_found": generic_dataset_subject_found,
        "missing_model_execution": missing_model_execution,
        "missing_dataset_execution": missing_dataset_execution,
        "simulation_hits": simulation_hits,
        "generic_fallback_source": fallback_source,
        "reasons": reasons,
        "checked_files": sorted(code_files),
        "generated_at": _utcnow_iso(),
    }


def _assess_pilot_envelope(
    files: dict[str, str],
    selected_topic: dict[str, Any],
) -> dict[str, Any]:
    """Fail closed when generated code expands a declared cheap pilot.

    Stage 10 may faithfully implement the scientific subject while still
    turning a one-GPU discriminating pilot into a full campaign.  Detect the
    most common hard-coded scale expansions before any sandbox/GPU execution.
    """

    cheap_pilot = str(selected_topic.get("cheap_pilot", "") or "").strip()
    compute = selected_topic.get("compute")
    if not isinstance(compute, dict):
        compute = {}
    declared_envelope = selected_topic.get("pilot_envelope")
    if not isinstance(declared_envelope, dict):
        declared_envelope = {}
    if not cheap_pilot and not compute and not declared_envelope:
        return {
            "aligned": True,
            "cheap_pilot_declared": False,
            "reasons": [],
            "observed": {},
        }

    combined = "\n\n".join(
        code for name, code in sorted(files.items()) if name.endswith(".py")
    )
    lowered = combined.casefold()

    def int_assignments(*names: str) -> list[int]:
        escaped = "|".join(re.escape(name) for name in names)
        return [
            int(match.group(1))
            for match in re.finditer(
                rf"(?mi)^\s*(?:{escaped})\s*(?::[^=\n]+)?=\s*(\d+)\b",
                combined,
            )
        ]

    seed_count = 0
    literal_assignments: dict[str, list[Any]] = {}
    for name, code in sorted(files.items()):
        if not name.endswith(".py"):
            continue
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            target_names: list[str] = []
            value: ast.AST | None = None
            if isinstance(node, ast.Assign):
                value = node.value
                target_names.extend(
                    target.id
                    for target in node.targets
                    if isinstance(target, ast.Name)
                )
            elif isinstance(node, ast.AnnAssign):
                value = node.value
                if isinstance(node.target, ast.Name):
                    target_names.append(node.target.id)
            if value is None:
                continue
            try:
                literal = ast.literal_eval(value)
            except (ValueError, TypeError):
                continue
            for target_name in target_names:
                literal_assignments.setdefault(
                    target_name.casefold(), []
                ).append(literal)

    for literal in literal_assignments.get("seeds", []):
        if isinstance(literal, (list, tuple, set)):
            seed_count = max(seed_count, len(literal))
    for match in re.finditer(
        r"(?mi)^\s*seeds\s*(?::[^=\n]+)?=\s*(?:field\([^=\n]*"
        r"default_factory\s*=\s*lambda:\s*)?\[([^\]]*)\]",
        combined,
    ):
        seed_count = max(
            seed_count,
            len(re.findall(r"(?<![\w.])-?\d+(?![\w.])", match.group(1))),
        )

    dict_example_keys = (
        "seed_size",
        "pool_size",
        "dev_size",
        "test_size",
        "num_examples",
        "max_examples",
        "pilot_examples",
        "sample_size",
        "prompts_per_round",
    )
    dict_iteration_keys = (
        "iterations",
        "num_iterations",
        "max_iterations",
        "pilot_rounds",
        "max_self_improvement_rounds",
        "rounds",
    )
    literal_examples: list[int] = []
    literal_iterations: list[int] = []
    for values in literal_assignments.values():
        for literal in values:
            if not isinstance(literal, dict):
                continue
            for raw_key, raw_value in literal.items():
                if isinstance(raw_value, bool) or not isinstance(
                    raw_value, int
                ):
                    continue
                key = str(raw_key).casefold()
                if key in dict_example_keys:
                    literal_examples.append(raw_value)
                if key in dict_iteration_keys:
                    literal_iterations.append(raw_value)

    proxy_markers_found = sorted(
        marker for marker in _LLM_PROXY_MARKERS if marker in lowered
    )

    observed = {
        "gpu_counts": int_assignments(
            "num_gpus_available",
            "num_gpus",
            "gpu_count",
            "max_gpu",
        ),
        "seed_count": seed_count,
        "iterations": int_assignments(
            "pilot_rounds",
            "max_self_improvement_rounds",
            "max_iterations",
            "num_iterations",
        ) + literal_iterations,
        "examples": int_assignments(
            "prompts_per_round",
            "num_examples",
            "max_examples",
            "pilot_examples",
            "sample_size",
            "seed_size",
            "pool_size",
            "dev_size",
            "test_size",
        ) + literal_examples,
        "proxy_markers_found": proxy_markers_found,
    }

    declared_gpu_count = compute.get("gpu_count", compute.get("max_gpu"))
    if declared_envelope.get("max_gpus") is not None:
        declared_gpu_count = declared_envelope.get("max_gpus")
    try:
        declared_gpu_limit = int(declared_gpu_count)
    except (TypeError, ValueError):
        declared_gpu_limit = None
    pilot_lower = cheap_pilot.casefold()
    if declared_gpu_limit is None and re.search(
        r"\b(?:one|1)[ -]?gpu\b", pilot_lower
    ):
        declared_gpu_limit = 1

    def optional_positive_int(key: str) -> int | None:
        value = declared_envelope.get(key)
        if value is None:
            return None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    declared_seed_limit = optional_positive_int("max_seeds")
    declared_iteration_limit = optional_positive_int("max_iterations")
    declared_example_limit = optional_positive_int("max_examples")

    reasons: list[str] = []
    if (
        declared_gpu_limit is not None
        and observed["gpu_counts"]
        and max(observed["gpu_counts"]) > declared_gpu_limit
    ):
        reasons.append(
            "generated code requests more GPUs than the selected topic's "
            f"cheap pilot ({max(observed['gpu_counts'])} > "
            f"{declared_gpu_limit})"
        )
    # These are conservative production-canary ceilings.  They allow the
    # selected topic's usual 3-5 round / 100-200 example pilot while blocking
    # accidental full-scale campaigns before Stage 11 can apply resources.
    seed_limit = declared_seed_limit or 3
    if observed["seed_count"] > seed_limit:
        reasons.append(
            "generated cheap-pilot code hard-codes more seeds than the "
            f"selected topic permits ({observed['seed_count']} > {seed_limit})"
        )
    iteration_limit = declared_iteration_limit or 5
    if (
        observed["iterations"]
        and max(observed["iterations"]) > iteration_limit
    ):
        reasons.append(
            "generated cheap-pilot code hard-codes more iterations than the "
            "selected topic permits "
            f"({max(observed['iterations'])} > {iteration_limit})"
        )
    example_limit = declared_example_limit or 200
    if observed["examples"] and max(observed["examples"]) > example_limit:
        reasons.append(
            "generated cheap-pilot code hard-codes more examples than the "
            f"selected topic permits ({max(observed['examples'])} > "
            f"{example_limit})"
        )
    contract_text = " ".join(
        _normalize_contract_values(selected_topic)
    ).casefold()
    if (
        any(marker in contract_text for marker in _LLM_SUBJECT_MARKERS)
        and proxy_markers_found
    ):
        reasons.append(
            "generated cheap pilot substitutes an image/small-model proxy "
            "for the declared LLM experiment"
        )

    return {
        "aligned": not reasons,
        "cheap_pilot_declared": True,
        "cheap_pilot": cheap_pilot,
        "declared_gpu_limit": declared_gpu_limit,
        "declared_seed_limit": declared_seed_limit,
        "declared_iteration_limit": declared_iteration_limit,
        "declared_example_limit": declared_example_limit,
        "observed": observed,
        "reasons": reasons,
    }


def _scientific_code_alignment_result(
    stage_dir: Path,
    report: dict[str, Any],
) -> StageResult | None:
    """Persist the Stage-10 scientific code gate and fail closed on drift."""

    report_path = stage_dir / "scientific_code_alignment.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if report.get("aligned", False):
        return None
    logger.error(
        "Stage 10 BLOCKED: generated code is not an authentic implementation "
        "of selected topic %r: %s",
        report.get("selected_topic_title"),
        "; ".join(str(reason) for reason in report.get("reasons", ())),
    )
    return StageResult(
        stage=Stage.CODE_GENERATION,
        status=StageStatus.PAUSED,
        artifacts=("experiment/", "scientific_code_alignment.json"),
        error=(
            "Generated code substitutes synthetic/simulated evidence for the "
            "authoritative selected-topic experiment"
        ),
        decision="scientific_code_misalignment",
        evidence_refs=("stage-10/scientific_code_alignment.json",),
    )


def _execute_collider_plan_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Stage 10 (collider_agent mode): generate a ColliderAgent physics prompt.

    Reads the experiment design plan from Stage 9 and uses the LLM to
    translate it into a detailed ColliderAgent-compatible Markdown prompt
    (similar to ``paper-reproduction/*/prompt_figure_N.md``).

    The generated prompt is saved as ``collider_plan.md`` in the stage
    directory.  Stage 12 reads this file and invokes Claude Code with the
    ColliderAgent skills to execute the full physics pipeline.
    """
    llm = _bind_stage_role(llm, Stage.CODE_GENERATION)
    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    hypothesis = _read_prior_artifact(run_dir, "hypotheses.json") or ""
    topic = config.research.topic

    # System prompt: instruct LLM to produce a ColliderAgent-style prompt
    system_prompt = (
        "You are a particle physics expert generating a detailed execution plan for "
        "the ColliderAgent framework. ColliderAgent uses Claude Code to orchestrate "
        "the full collider phenomenology pipeline:\n"
        "  1. FeynRules model generation from a Lagrangian\n"
        "  2. UFO export for MadGraph5\n"
        "  3. MadGraph5 event generation with Pythia8/Delphes\n"
        "  4. MadAnalysis5 analysis\n"
        "  5. Numerical post-processing and figure generation\n\n"
        "The execution plan must follow this Markdown structure:\n"
        "  # 1. Target\n"
        "  (what figure/result to produce)\n"
        "  # 2. Model\n"
        "  ## 2.1 Lagrangian\n"
        "  ## 2.2 Parameters\n"
        "  ## 2.3 Particles\n"
        "  # 3. Collider Process\n"
        "  ## 3.1 Signal Process\n"
        "  ## 3.2 Background Process (if any)\n"
        "  # 4. Numerical Analysis\n"
        "  (step-by-step procedure)\n\n"
        "Be as precise as possible with formulas, parameter values, and analysis steps. "
        "If the topic does not have a defined Lagrangian or specific HEP process, "
        "generate an equivalent phenomenological study appropriate for the topic. "
        "If MadGraph/Monte Carlo is not needed (pure numerical analysis), skip those steps "
        "and describe only the post-processing steps."
    )
    user_prompt = (
        f"Research topic: {topic}\n\n"
        f"Experiment design plan:\n{exp_plan}\n\n"
        f"Hypotheses:\n{hypothesis}\n\n"
        "Generate a detailed ColliderAgent execution plan as a Markdown document."
    )

    collider_plan: str
    if llm is not None:
        try:
            resp = _chat_with_prompt(llm, system_prompt, user_prompt, max_tokens=4096)
            collider_plan = resp.content.strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Stage 10 (collider_agent): LLM call failed (%s) — using fallback plan", exc)
            collider_plan = _fallback_collider_plan(topic, exp_plan)
    else:
        collider_plan = _fallback_collider_plan(topic, exp_plan)

    # Write the plan
    plan_path = stage_dir / "collider_plan.md"
    plan_path.write_text(collider_plan, encoding="utf-8")
    logger.info("Stage 10 (collider_agent): wrote physics prompt to %s", plan_path)

    # Also write a metadata file
    import json as _json
    meta = {
        "generated": _utcnow_iso(),
        "mode": "collider_agent",
        "topic": topic,
        "plan_file": "collider_plan.md",
        "plan_length_chars": len(collider_plan),
    }
    (stage_dir / "collider_meta.json").write_text(
        _json.dumps(meta, indent=2), encoding="utf-8"
    )

    # Satisfy Stage 10 contract (output_files requires "experiment/" and
    # "experiment_spec.md").  In collider_agent mode there is no Python
    # experiment; instead we place the ColliderAgent prompt inside the
    # experiment/ directory so downstream contract validation passes.
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(exist_ok=True)
    (exp_dir / "collider_plan.md").write_text(collider_plan, encoding="utf-8")

    spec_md = (
        f"# Experiment Specification (collider_agent mode)\n\n"
        f"**Topic:** {topic}\n\n"
        f"**Backend:** ColliderAgent — full HEP pipeline via Claude Code\n\n"
        f"**Physics plan:** `collider_plan.md`\n\n"
        f"Stage 12 will invoke the Claude Code CLI with the ColliderAgent skills\n"
        f"to execute the Lagrangian → FeynRules → UFO → MadGraph5 → Delphes →\n"
        f"MadAnalysis5 pipeline and produce publication-quality figures.\n"
    )
    (stage_dir / "experiment_spec.md").write_text(spec_md, encoding="utf-8")

    return StageResult(
        stage=Stage.CODE_GENERATION,
        status=StageStatus.DONE,
        artifacts=("collider_plan.md", "collider_meta.json", "experiment/", "experiment_spec.md"),
        evidence_refs=("stage-10/collider_plan.md",),
    )


def _fallback_collider_plan(topic: str, exp_plan: str) -> str:
    """Generate a minimal fallback ColliderAgent prompt when LLM is unavailable."""
    return f"""# 1. Target

Investigate the following physics topic using the ColliderAgent pipeline:
**{topic}**

{exp_plan or "Execute the relevant collider phenomenology analysis and generate exclusion contours or kinematic distributions as appropriate."}

---

# 2. Model

## 2.1 Lagrangian

Use the Standard Model as baseline. For beyond-SM contributions,
refer to the experiment design plan above.

## 2.2 Parameters

Use SM parameters. Scan over new-physics parameters as described in the plan.

## 2.3 Particles

Use standard SM particles.

---

# 3. Collider Process

## 3.1 Signal Process

Run the relevant signal processes at the LHC (√s = 13 TeV).

---

# 4. Numerical Analysis

## Step 1: Execute the phenomenology pipeline
Follow the experiment design plan to produce the required figures and results.

## Step 2: Generate output figures
Save all figures to output/figures/ in PDF and PNG format.
"""


def _check_rl_compatibility(code: str) -> list[str]:
    """Detect DQN + continuous-action environment mismatches.

    Returns a list of error strings if incompatible combinations are found.
    """
    errors: list[str] = []
    code_lower = code.lower()
    has_dqn = "dqn" in code_lower
    if not has_dqn:
        return errors

    for env_name in _CONTINUOUS_ENVS:
        if env_name in code_lower:
            errors.append(
                f"RL COMPATIBILITY ERROR: DQN is used with continuous-action "
                f"environment '{env_name}'. DQN only works with DISCRETE action "
                f"spaces. Use SAC, TD3, or PPO instead."
            )
    return errors


def _execute_code_generation(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    llm = _bind_stage_role(llm, Stage.CODE_GENERATION)
    # ── ColliderAgent mode: generate a physics prompt instead of Python code ─
    if config.experiment.mode == "collider_agent":
        return _execute_collider_plan_generation(
            stage_dir, run_dir, config, adapters, llm=llm, prompts=prompts
        )
    # ── End ColliderAgent bypass ──────────────────────────────────────────────

    exp_plan = _read_prior_artifact(run_dir, "exp_plan.yaml") or ""
    metric = config.experiment.metric_key
    max_repair = 5  # BUG-14: Increased from 3 to give more chances for critical bugs
    files: dict[str, str] = {}
    validation_log: list[str] = []

    # --- Detect available packages for sandbox ---
    _pm = prompts or PromptManager()
    _scientific_contract = _load_scientific_contract(run_dir, config)
    _compiled_contract = _implementation_contract(
        _scientific_contract,
        config.research.topic,
    )
    _authoritative_topic = str(
        _compiled_contract.get("title", config.research.topic)
        or config.research.topic
    )
    (stage_dir / "implementation_contract.json").write_text(
        json.dumps(
            _compiled_contract,
            indent=2,
            ensure_ascii=False,
        ) + "\n",
        encoding="utf-8",
    )
    _implementation_manifest: dict[str, Any] = {}
    if _compiled_contract.get("topic_id"):
        if llm is None:
            return StageResult(
                stage=Stage.CODE_GENERATION,
                status=StageStatus.PAUSED,
                artifacts=("implementation_contract.json",),
                error=(
                    "Authoritative topic requires an approved implementation "
                    "manifest before code generation, but no LLM is available"
                ),
                decision="implementation_manifest_required",
                evidence_refs=("stage-10/implementation_contract.json",),
            )
        (
            _implementation_manifest,
            _manifest_report,
        ) = _prepare_implementation_manifest(
            stage_dir,
            _compiled_contract,
            exp_plan,
            llm,
        )
        if not _manifest_report.get("approved", False):
            return StageResult(
                stage=Stage.CODE_GENERATION,
                status=StageStatus.PAUSED,
                artifacts=(
                    "implementation_contract.json",
                    "implementation_manifest.json",
                    "implementation_manifest_gate.json",
                ),
                error=(
                    "Implementation manifest violates the authoritative "
                    "scientific contract"
                ),
                decision="implementation_manifest_rejected",
                evidence_refs=(
                    "stage-10/implementation_contract.json",
                    "stage-10/implementation_manifest_gate.json",
                ),
            )
        _manifest_guidance = _implementation_guidance(
            _compiled_contract,
            _implementation_manifest,
        )
    else:
        _manifest_guidance = ""

    # --- Hardware-aware package hint ---
    hw_profile = _load_hardware_profile(run_dir)
    if config.experiment.mode in (
        "sandbox",
        "docker",
        "clusterbridge",
        "clusterbridge_pool",
    ):
        if config.experiment.mode == "docker":
            pkg_prefix = "docker mode"
            _net_policy = config.experiment.docker.network_policy
            _base_pkgs = (
                ", torchvision, torchaudio, matplotlib, seaborn, scipy, "
                "tqdm, torchdiffeq, gymnasium, networkx, PyYAML, Pillow, "
                "transformers, datasets, accelerate, peft, bitsandbytes, "
                "timm, einops, torchmetrics, h5py"
            )
            if _net_policy == "none":
                pkg_extras = _base_pkgs + " (ONLY pre-installed packages — NO pip install available)"
            elif _net_policy in ("setup_only", "pip_only"):
                pkg_extras = _base_pkgs + ", and additional pip-installable packages via requirements.txt"
            else:
                pkg_extras = _base_pkgs + ", and additional pip-installable packages (auto-detected from imports)"
        elif config.experiment.mode in ("clusterbridge", "clusterbridge_pool"):
            pkg_prefix = (
                "clusterbridge multi-node Ray GPU pool mode"
                if config.experiment.mode == "clusterbridge_pool"
                else "clusterbridge remote GPU mode"
            )
            pkg_extras = (
                ", torchvision, torchaudio, matplotlib, scipy, pandas, tqdm, "
                "transformers, datasets, accelerate, scikit-learn"
            )
        else:
            pkg_prefix = "sandbox mode"
            pkg_extras = ""
        if hw_profile and hw_profile.get("has_gpu"):
            gpu_type = hw_profile.get("gpu_type", "cuda")
            gpu_name = hw_profile.get("gpu_name", "GPU")
            tier = hw_profile.get("tier", "limited")
            if tier == "high":
                device_hint = f"torch.device('{gpu_type}')"
                pkg_hint = (
                    f"\nAVAILABLE PACKAGES ({pkg_prefix}): Python stdlib, numpy, torch, sklearn, scipy, pandas{pkg_extras}.\n"
                    f"GPU: {gpu_name} ({gpu_type}). You MAY use PyTorch with GPU acceleration.\n"
                    f"Use `device = {device_hint}` for tensor operations.\n"
                )
            else:  # limited (low VRAM NVIDIA or MPS)
                device_hint = f"torch.device('{gpu_type}')"
                pkg_hint = (
                    f"\nAVAILABLE PACKAGES ({pkg_prefix}): Python stdlib, numpy, torch, sklearn, scipy, pandas{pkg_extras}.\n"
                    f"GPU: {gpu_name} ({gpu_type}) — LIMITED performance.\n"
                    f"Use `device = {device_hint}` but design LIGHTWEIGHT experiments:\n"
                    f"- Small models (<1M parameters)\n"
                    f"- Few epochs (<=20)\n"
                    f"- Small datasets (<=10K samples)\n"
                    f"- Avoid large batch sizes\n"
                )
        else:
            pkg_hint = _pm.block("pkg_hint_sandbox")
    else:
        pkg_hint = ""

    # --- Compute budget hint ---
    time_budget_sec = config.experiment.time_budget_sec
    try:
        compute_budget = _pm.block("compute_budget").replace(
            "{time_budget_sec}", str(time_budget_sec)
        )
    except Exception:  # noqa: BLE001
        compute_budget = (
            f"\n## Compute Budget Constraint\n"
            f"- Total execution time limit: {time_budget_sec} seconds\n"
            f"- Design experiments that complete within this budget\n"
            f"- Implement a time guard: stop gracefully at 80% of budget\n"
        )

    # --- Dataset guidance + setup script + HP reporting (docker/sandbox modes) ---
    extra_guidance = ""
    _net_policy = getattr(getattr(config, "docker", None), "network_policy", "setup_only")
    if config.experiment.mode in (
        "sandbox",
        "docker",
        "clusterbridge",
        "clusterbridge_pool",
    ):
        _net_policy = (
            config.experiment.docker.network_policy
            if config.experiment.mode == "docker"
            else "none"  # sandbox/clusterbridge experiment phase has no network
        )
        if _net_policy == "none":
            # Network disabled: inject strict offline-only guidance
            if _compiled_contract.get("topic_id"):
                extra_guidance += (
                    "\n## OFFLINE RUNTIME CONTRACT\n"
                    "Stage 12 has no public network. The exact approved model "
                    "checkpoint and benchmark must therefore be pre-cached or "
                    "staged onto the worker before execution. Do NOT substitute "
                    "an unrelated pre-cached dataset or toy model. Fail clearly "
                    "when the approved artifacts are unavailable.\n"
                )
            else:
                try:
                    extra_guidance += _pm.block(
                        "network_disabled_guidance"
                    )
                except Exception:  # noqa: BLE001
                    pass
        elif _net_policy == "full":
            try:
                extra_guidance += _pm.block("dataset_guidance")
                extra_guidance += _pm.block("network_full_guidance")
            except Exception:  # noqa: BLE001
                pass
        else:
            # setup_only or pip_only — existing behavior
            try:
                extra_guidance += _pm.block("dataset_guidance")
            except Exception:  # noqa: BLE001
                pass
            if config.experiment.mode == "docker":
                try:
                    extra_guidance += _pm.block("setup_script_guidance")
                except Exception:  # noqa: BLE001
                    pass
        try:
            extra_guidance += _pm.block("hp_reporting")
        except Exception:  # noqa: BLE001
            pass
        # I-06: Multi-seed defaults apply only when the authoritative pilot
        # contract has not selected a stricter seed envelope.
        _seed_limit = _compiled_contract.get("pilot_envelope", {}).get(
            "max_seeds"
        )
        if _seed_limit is None:
            try:
                extra_guidance += _pm.block("multi_seed_enforcement")
            except Exception:  # noqa: BLE001
                pass
        else:
            extra_guidance += (
                "\n## PILOT SEED LIMIT (AUTHORITATIVE)\n"
                f"Use exactly {_seed_limit} seed(s) in this canary. "
                "Do not expand to the generic three-seed paper protocol until "
                "the pilot is explicitly promoted.\n"
            )

    # --- BA: Inject BenchmarkAgent plan from Stage 9 ---
    _bp_path = None
    for _s9_dir in sorted(run_dir.glob("stage-09*"), reverse=True):
        _candidate = _s9_dir / "benchmark_plan.json"
        if _candidate.exists():
            _bp_path = _candidate
            break
    if _bp_path is not None:
        try:
            import json as _json_bp
            _bp_data = _json_bp.loads(_bp_path.read_text(encoding="utf-8"))
            # Reconstruct the prompt block
            from researchclaw.agents.benchmark_agent.orchestrator import BenchmarkPlan
            _bp = BenchmarkPlan(
                selected_benchmarks=_bp_data.get("selected_benchmarks", []),
                selected_baselines=_bp_data.get("selected_baselines", []),
                data_loader_code=_bp_data.get("data_loader_code", ""),
                baseline_code=_bp_data.get("baseline_code", ""),
                experiment_notes=_bp_data.get("experiment_notes", ""),
            )
            _bp_block = _bp.to_prompt_block()
            if _bp_block:
                extra_guidance += (
                    "\n\n## BenchmarkAgent Selections (USE THESE)\n"
                    "The following datasets, baselines, and code snippets were "
                    "automatically selected and validated by the BenchmarkAgent. "
                    "You MUST use these selections in your experiment code.\n\n"
                    + _bp_block
                )
                logger.info(
                    "BA: Injected benchmark plan (%d benchmarks, %d baselines)",
                    len(_bp.selected_benchmarks), len(_bp.selected_baselines),
                )
        except Exception as _bp_exc:
            logger.debug("BA: Failed to load benchmark plan: %s", _bp_exc)

    # --- P2.2+P2.3: LLM training topic detection and guidance ---
    _llm_keywords = (
        "language model", "llm", "fine-tun", "lora", "qlora", "peft",
        "instruction tun", "rlhf", "dpo", "sft", "alignment",
        "transformer train", "causal lm", "chat model", "qwen", "llama",
        "mistral", "phi-", "gemma", "pretraining", "tokeniz",
    )
    topic_lower = _authoritative_topic.lower()
    is_llm_topic = any(kw in topic_lower for kw in _llm_keywords)

    # --- I-08: RL topic detection and step guidance ---
    _rl_keywords = (
        "reinforcement learning", "policy gradient", "ppo", "sac", "td3",
        "ddpg", "dqn", "a2c", "a3c", "mujoco", "locomotion", "continuous control",
        "reward shaping", "exploration", "multi-agent rl", "marl", "curriculum rl",
        "imitation learning", "inverse rl", "offline rl", "model-based rl",
        "actor-critic", "reinforce", "gym", "gymnasium",
    )
    is_rl_topic = any(kw in topic_lower for kw in _rl_keywords)
    if is_rl_topic:
        try:
            extra_guidance += _pm.block("rl_step_guidance")
        except Exception:  # noqa: BLE001
            pass

    # --- F-01: Framework API doc injection (auto-detected) ---
    try:
        from researchclaw.data import detect_frameworks, load_framework_docs
        _hypothesis_text = _read_prior_artifact(run_dir, "hypotheses.md") or ""
        _fw_ids = detect_frameworks(
            _authoritative_topic, _hypothesis_text, exp_plan or ""
        )
        if _fw_ids:
            _fw_docs = load_framework_docs(_fw_ids, max_chars=8000)
            if _fw_docs:
                extra_guidance += _fw_docs
                logger.info("F-01: Injected framework docs for: %s", _fw_ids)
    except Exception:  # noqa: BLE001
        logger.debug("F-01: Framework doc injection skipped", exc_info=True)

    if is_llm_topic and config.experiment.mode == "docker":
        try:
            extra_guidance += _pm.block("llm_training_guidance")
        except Exception:  # noqa: BLE001
            pass
        try:
            extra_guidance += _pm.block("llm_eval_guidance")
        except Exception:  # noqa: BLE001
            pass
        # P2.3: Warn if time budget is too short for LLM training
        if time_budget_sec < 3600:
            extra_guidance += (
                "\n## COMPUTE BUDGET WARNING\n"
                f"Current time_budget_sec={time_budget_sec} is likely TOO SHORT "
                f"for LLM fine-tuning. Typical LoRA training needs 1-4 hours. "
                f"Design a LIGHTWEIGHT experiment:\n"
                f"- Use a small dataset (<=5000 samples)\n"
                f"- Train for 1-3 epochs only\n"
                f"- Use small batch size (1-2) with gradient accumulation\n"
                f"- Use 4-bit quantization (QLoRA) to minimize memory\n"
                f"- Limit max_seq_length to 512-1024\n"
                f"- If possible, use a smaller model (<=7B parameters)\n"
            )

    # --- Domain-specific guidance injection for non-ML domains ---
    try:
        from researchclaw.domains.detector import detect_domain as _dd_s10, is_ml_domain as _is_ml_s10
        _dp = _dd_s10(topic=_authoritative_topic)
        if not _is_ml_s10(_dp):
            from researchclaw.domains.prompt_adapter import get_adapter as _ga
            _adapter = _ga(_dp)
            _blocks = _adapter.get_code_generation_blocks({})
            if _blocks.compute_budget:
                compute_budget = _blocks.compute_budget
            if _blocks.dataset_guidance:
                extra_guidance = _blocks.dataset_guidance + "\n" + extra_guidance
            if _blocks.code_generation_hints:
                extra_guidance += "\n" + _blocks.code_generation_hints
            if _blocks.output_format_guidance:
                extra_guidance += "\n" + _blocks.output_format_guidance
            logger.info("Injected domain-specific guidance for %s", _dp.domain_id)
    except Exception:  # noqa: BLE001
        logger.debug("Domain guidance injection skipped", exc_info=True)

    # BUG-R6-01: Add explicit implementation constraints to prevent LLM
    # from substituting unrelated DL models for lightweight algorithms.
    _requires_gpu_subject = bool(
        _compiled_contract.get("required_capabilities", {}).get(
            "llm_subject",
            False,
        )
    )
    extra_guidance += (
        "\n\nIMPLEMENTATION CONSTRAINTS (MUST FOLLOW):\n"
        "- Implement EXACTLY the algorithm/method described in the topic.\n"
        "- Do NOT replace the stated method with a deep-learning proxy "
        "(e.g. ResNet, BERT, GPT, Gymnasium+SB3) unless the topic "
        "EXPLICITLY requires deep learning.\n"
        "- Prefer lightweight CPU-friendly libraries (numpy, scipy, "
        "sklearn, pandas) unless deep learning is inherent to the topic.\n"
        + (
            "- Deep learning is inherent to this authoritative topic; use the "
            "approved real model and the bounded GPU envelope.\n"
            if _requires_gpu_subject
            else "- The experiment MUST be self-contained and runnable without GPU.\n"
        )
        + _manifest_guidance
    )

    # --- Code generation: Beast Mode → CodeAgent → Legacy single-shot ---
    _code_agent_active = False
    _beast_mode_used = False
    _code_max_tokens = 8192

    # ── Beast Mode: OpenCode external agent (optional) ─────────────────
    _oc_cfg = config.experiment.opencode
    if _oc_cfg.enabled:
        from researchclaw.pipeline.opencode_bridge import (
            OpenCodeBridge,
            OpenCodeResult,
            count_historical_failures,
            score_complexity,
        )

        _hist_failures = count_historical_failures(run_dir)
        _cplx = score_complexity(
            exp_plan=exp_plan,
            topic=_authoritative_topic,
            historical_failures=_hist_failures,
            threshold=_oc_cfg.complexity_threshold,
        )

        # Persist complexity analysis
        (stage_dir / "complexity_analysis.json").write_text(
            json.dumps(
                {
                    "score": _cplx.score,
                    "signals": _cplx.signals,
                    "recommendation": _cplx.recommendation,
                    "reason": _cplx.reason,
                    "threshold": _oc_cfg.complexity_threshold,
                    "historical_failures": _hist_failures,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if _cplx.recommendation == "beast_mode":
            _proceed = _oc_cfg.auto
            if not _proceed:
                # Non-auto mode: check for HITL adapter
                if adapters.hitl is not None:
                    try:
                        _proceed = adapters.hitl.confirm(
                            f"Beast Mode: complexity={_cplx.score:.2f} "
                            f"(threshold={_oc_cfg.complexity_threshold}). "
                            f"Route to OpenCode?"
                        )
                    except Exception:  # noqa: BLE001
                        logger.info(
                            "Beast mode: HITL adapter unavailable, skipping "
                            "(set opencode.auto=true for non-interactive runs)"
                        )
                else:
                    logger.info(
                        "Beast mode: no HITL adapter, skipping "
                        "(set opencode.auto=true for non-interactive runs)"
                    )

            if _proceed:
                _oc_model = _oc_cfg.model or config.llm.primary_model
                _bridge = OpenCodeBridge(
                    model=_oc_model,
                    llm_base_url=config.llm.base_url,
                    api_key_env=config.llm.api_key_env,
                    llm_provider=config.llm.provider,
                    timeout_sec=_oc_cfg.timeout_sec,
                    max_retries=_oc_cfg.max_retries,
                    workspace_cleanup=_oc_cfg.workspace_cleanup,
                )

                logger.info(
                    "Beast mode: ENGAGED (complexity=%.2f, model=%s)",
                    _cplx.score,
                    _oc_model,
                )

                _oc_result: OpenCodeResult = _bridge.generate(
                    stage_dir=stage_dir,
                    topic=_authoritative_topic,
                    exp_plan=exp_plan,
                    metric=metric,
                    pkg_hint=pkg_hint + "\n" + compute_budget,
                    extra_guidance=extra_guidance,
                    time_budget_sec=config.experiment.time_budget_sec,
                )

                # Persist beast mode log
                (stage_dir / "beast_mode_log.json").write_text(
                    json.dumps(
                        {
                            "success": _oc_result.success,
                            "elapsed_sec": _oc_result.elapsed_sec,
                            "files": list(_oc_result.files.keys()),
                            "error": _oc_result.error,
                            "complexity_score": _cplx.score,
                            "model": _oc_model,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )

                if _oc_result.success and _oc_result.files:
                    files = _oc_result.files
                    _beast_mode_used = True
                    _code_agent_active = True  # skip legacy path
                    logger.info(
                        "Beast mode: SUCCESS — %d files in %.1fs",
                        len(files),
                        _oc_result.elapsed_sec,
                    )
                else:
                    logger.warning(
                        "Beast mode: FAILED (%s) — falling back to CodeAgent",
                        _oc_result.error or "unknown error",
                    )
        else:
            logger.info(
                "Beast mode: complexity=%.2f (threshold=%.2f), not triggered",
                _cplx.score,
                _oc_cfg.complexity_threshold,
            )

    if not _beast_mode_used and config.experiment.code_agent.enabled and llm is not None:
        # ── F-02: Advanced Code Agent path ────────────────────────────────
        from researchclaw.pipeline.code_agent import CodeAgent as _CodeAgent

        _ca_cfg = config.experiment.code_agent
        # Ensure we have a proper config object
        if not hasattr(_ca_cfg, "enabled"):
            from researchclaw.pipeline.code_agent import (
                CodeAgentConfig as _CAConfig,
            )
            _ca_cfg = _CAConfig()

        # Sandbox factory (only for sandbox/docker modes)
        _sandbox_factory = None
        if config.experiment.mode in (
            "sandbox",
            "docker",
            "clusterbridge",
            "clusterbridge_pool",
        ):
            from researchclaw.experiment.factory import (
                create_sandbox as _csb,
            )
            _sandbox_factory = _csb

        if any(
            config.llm.primary_model.startswith(p)
            for p in ("gpt-5", "o3", "o4")
        ):
            _code_max_tokens = 16384

        # ── Domain detection + Code Search for non-ML domains ──────────
        _domain_profile = None
        _code_search_result = None
        try:
            from researchclaw.domains.detector import detect_domain as _dd
            from researchclaw.domains.detector import is_ml_domain as _is_ml
            _domain_profile = _dd(topic=_authoritative_topic)
            logger.info(
                "CodeAgent: domain=%s (%s)",
                _domain_profile.display_name,
                _domain_profile.domain_id,
            )
            # Run code search for non-ML domains (ML has enough built-in knowledge)
            if not _is_ml(_domain_profile):
                try:
                    from researchclaw.agents.code_searcher import CodeSearchAgent
                    _cs_agent = CodeSearchAgent(llm=llm)
                    _code_search_result = _cs_agent.search(
                        topic=_authoritative_topic,
                        domain=_domain_profile,
                    )
                    if _code_search_result and _code_search_result.patterns.has_content:
                        logger.info(
                            "Code search: %d patterns, %d repos found",
                            len(_code_search_result.patterns.api_patterns),
                            len(_code_search_result.repos_found),
                        )
                except Exception:  # noqa: BLE001
                    logger.debug("Code search unavailable", exc_info=True)
        except Exception:  # noqa: BLE001
            logger.debug("Domain detection unavailable", exc_info=True)

        _agent = _CodeAgent(
            llm=llm,
            prompts=_pm,
            config=_ca_cfg,
            stage_dir=stage_dir,
            # CodeAgent execution-in-the-loop must not launch a real remote
            # cluster task before Stage 10's scientific and pilot gates have
            # accepted the generated project.  Stage 12 remains the only
            # production experiment execution boundary for cluster modes.
            sandbox_factory=(
                None
                if config.experiment.mode in ("clusterbridge", "clusterbridge_pool")
                else _sandbox_factory
            ),
            experiment_config=config.experiment,
            domain_profile=_domain_profile,
            code_search_result=_code_search_result,
        )
        _campaign_code_guidance = _pm.for_stage(
            "code_generation",
            evolution_overlay=_get_evolution_overlay(
                run_dir,
                "code_generation",
                config=config,
                topic=_authoritative_topic,
            ),
            topic=_authoritative_topic,
            metric=metric,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            exp_plan=exp_plan,
            metric_direction_hint="",
        ).user
        _agent_result = _agent.generate(
            topic=_authoritative_topic,
            exp_plan=exp_plan,
            metric=metric,
            pkg_hint=(
                pkg_hint
                + "\n"
                + compute_budget
                + "\n"
                + extra_guidance
                + "\n\n## Campaign and Evolution Guidance\n"
                + _campaign_code_guidance
            ),
            max_tokens=_code_max_tokens,
        )
        files = _agent_result.files
        _code_agent_active = True

        # Write agent artifacts
        (stage_dir / "code_agent_log.json").write_text(
            json.dumps(
                {
                    "log": _agent_result.validation_log,
                    "llm_calls": _agent_result.total_llm_calls,
                    "sandbox_runs": _agent_result.total_sandbox_runs,
                    "best_score": _agent_result.best_score,
                    "tree_nodes_explored": _agent_result.tree_nodes_explored,
                    "review_rounds": _agent_result.review_rounds,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        if _agent_result.architecture_spec:
            (stage_dir / "architecture_spec.yaml").write_text(
                _agent_result.architecture_spec, encoding="utf-8",
            )
        logger.info(
            "CodeAgent: %d LLM calls, %d sandbox runs, score=%.2f",
            _agent_result.total_llm_calls,
            _agent_result.total_sandbox_runs,
            _agent_result.best_score,
        )
    elif not _beast_mode_used and llm is not None:
        # ── Legacy single-shot generation ─────────────────────────────────
        topic = _authoritative_topic
        _md = config.experiment.metric_direction
        _md_hint = (
            f"`{_md}` — use direction={'lower' if _md == 'minimize' else 'higher'} "
            f"in METRIC_DEF. You MUST NOT use the opposite direction."
        )
        _overlay = _get_evolution_overlay(run_dir, "code_generation")
        sp = _pm.for_stage(
            "code_generation",
            evolution_overlay=_overlay,
            topic=topic,
            metric=metric,
            pkg_hint=pkg_hint + "\n" + compute_budget + "\n" + extra_guidance,
            exp_plan=exp_plan,
            metric_direction_hint=_md_hint,
        )
        # R13-3: Use higher max_tokens for reasoning models (they consume tokens
        # for internal chain-of-thought). Retry once with even higher limit on empty.
        _code_max_tokens = sp.max_tokens or 8192
        if any(config.llm.primary_model.startswith(p) for p in ("gpt-5", "o3", "o4")):
            _code_max_tokens = max(_code_max_tokens, 16384)

        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=_code_max_tokens,
        )
        files = _extract_multi_file_blocks(resp.content)
        if not files and not resp.content.strip():
            # Empty response — retry with higher token limit
            logger.warning(
                "R13-3: Empty LLM response for code_generation (len=%d, "
                "finish_reason=%s, tokens=%d). Retrying with 32768 tokens.",
                len(resp.content),
                resp.finish_reason,
                resp.total_tokens,
            )
            resp = _chat_with_prompt(
                llm,
                sp.system,
                sp.user,
                json_mode=sp.json_mode,
                max_tokens=32768,
            )
            files = _extract_multi_file_blocks(resp.content)
        if not files:
            logger.warning(
                "R13-2: _extract_multi_file_blocks returned empty. "
                "LLM response length=%d, first 300 chars: %s",
                len(resp.content),
                resp.content[:300],
            )

    # --- Fallback: generic numerical experiment ---
    if not files:
        files = {
            "main.py": (
                "import numpy as np\n"
                "\n"
                "np.random.seed(42)\n"
                "\n"
                "# Fallback experiment: parameter sweep on a synthetic objective\n"
                "# This runs when LLM code generation fails to produce valid code.\n"
                "dim = 10\n"
                "n_conditions = 3\n"
                "results = {}\n"
                "\n"
                "for cond_idx in range(n_conditions):\n"
                "    cond_name = f'condition_{cond_idx}'\n"
                "    scores = []\n"
                "    for seed in range(3):\n"
                "        rng = np.random.RandomState(seed + cond_idx * 100)\n"
                "        x = rng.randn(dim)\n"
                "        score = float(1.0 / (1.0 + np.sum(x ** 2)))\n"
                "        scores.append(score)\n"
                "    mean_score = float(np.mean(scores))\n"
                "    results[cond_name] = mean_score\n"
                f"    print(f'condition={{cond_name}} {metric}: {{mean_score:.6f}}')\n"
                "\n"
                "best = max(results, key=results.get)\n"
                f"print(f'{metric}: {{results[best]:.6f}}')\n"
            )
        }

    # --- Validate each file + auto-repair loop ---
    all_valid = True
    attempt = 0
    for fname, code in list(files.items()):
        # Skip non-Python files (requirements.txt, setup.py, etc.)
        if not fname.endswith(".py"):
            continue
        validation = validate_code(code)
        repair_attempt = 0
        while not validation.ok and llm is not None and repair_attempt < max_repair:
            repair_attempt += 1
            attempt += 1
            # Only send errors to the LLM — warnings don't block validation
            # and confuse the LLM into over-correcting (e.g. removing runtime imports)
            errors_only = type(validation)(
                issues=[i for i in validation.issues if i.severity == "error"]
            )
            issues_text = format_issues_for_llm(errors_only)
            validation_log.append(
                f"File {fname} attempt {repair_attempt}: {validation.summary()}"
            )
            logger.info(
                "Code validation failed for %s (attempt %d/%d): %s",
                fname,
                repair_attempt,
                max_repair,
                validation.summary(),
            )
            all_files_ctx = "\n\n".join(
                f"```filename:{f}\n{c}\n```" for f, c in files.items()
            )
            rp = _pm.sub_prompt(
                "code_repair",
                fname=fname,
                issues_text=issues_text,
                all_files_ctx=all_files_ctx,
            )
            resp = _chat_with_prompt(llm, rp.system, rp.user)
            _repaired = _extract_code_block(resp.content)
            if _repaired.strip():
                files[fname] = _repaired
            else:
                logger.warning("Repair attempt returned empty code, keeping original")
            validation = validate_code(files[fname])
        if not validation.ok:
            all_valid = False
            # BUG-14: Log remaining issues prominently
            logger.warning(
                "Code validation FAILED for %s after %d repair attempts: %s",
                fname, max_repair, validation.summary(),
            )

    # Improvement G: RL algorithm-environment compatibility check
    for fname, code in list(files.items()):
        if not fname.endswith(".py"):
            continue
        _rl_errors = _check_rl_compatibility(code)
        if _rl_errors:
            for _rl_err in _rl_errors:
                logger.error("Stage 10: %s (in %s)", _rl_err, fname)
                validation_log.append(f"RL_COMPAT: {fname}: {_rl_err}")
            all_valid = False

    # BUG-14: Block on critical validation failures (syntax/import errors)
    if not all_valid:
        _has_critical = False
        for fname, code in files.items():
            _v = validate_code(code)
            if not _v.ok:
                for issue in _v.issues:
                    if issue.severity == "error" and issue.category in (
                        "syntax", "import",
                    ):
                        _has_critical = True
        if _has_critical:
            logger.error(
                "Stage 10: CRITICAL validation issues remain after %d repair "
                "attempts. Blocking stage.", max_repair,
            )
            (stage_dir / "validation_report.md").write_text(
                "# Code Validation Report\n\n"
                f"**Status**: BLOCKED — critical issues remain after {max_repair} repairs\n\n"
                + "\n".join(f"- {e}" for e in validation_log),
                encoding="utf-8",
            )
            return StageResult(
                stage=Stage.CODE_GENERATION,
                status=StageStatus.FAILED,
                artifacts=("validation_report.md",),
                error=(
                    "Critical generated-code validation issues remain after "
                    f"{max_repair} repair attempts"
                ),
                decision="code_validation_failed",
                evidence_refs=(),
            )

    # --- BUG-184: Cross-import validation — warn if a .py file imports a
    # local module that doesn't exist in the files dict.  This catches the
    # case where Beast Mode/CodeAgent produced an intermediate file that
    # got lost during repair iterations.
    for fname, module in _missing_generated_imports(files):
        logger.warning(
            "BUG-184: %s imports '%s' which is not in generated "
            "files — experiment may crash on import",
            fname, module,
        )

    # --- Write experiment directory ---
    exp_dir = stage_dir / "experiment"
    exp_dir.mkdir(parents=True, exist_ok=True)
    for fname, code in files.items():
        (exp_dir / fname).write_text(code, encoding="utf-8")

    # --- Write validation report ---
    if validation_log or not all_valid:
        report_lines = ["# Code Validation Report\n"]
        if all_valid:
            report_lines.append(f"**Status**: PASSED after {attempt} total repair(s)\n")
        else:
            report_lines.append(
                f"**Status**: FAILED after {attempt} total repair attempt(s)\n"
            )
        for entry in validation_log:
            report_lines.append(f"- {entry}")
        (stage_dir / "validation_report.md").write_text(
            "\n".join(report_lines), encoding="utf-8"
        )

    # --- R10-Fix6: Code complexity and quality check ---
    from researchclaw.experiment.validator import (
        auto_fix_unbound_locals,
        check_code_complexity,
        deep_validate_files,
    )

    # --- BUG-3 fix: Programmatic auto-fix for UnboundLocalError patterns ---
    _total_ub_fixes = 0
    for fname, code in list(files.items()):
        if fname.endswith(".py"):
            fixed_code, n_fixes = auto_fix_unbound_locals(code)
            if n_fixes > 0:
                files[fname] = fixed_code
                (exp_dir / fname).write_text(fixed_code, encoding="utf-8")
                _total_ub_fixes += n_fixes
                logger.info(
                    "Stage 10: auto-fixed %d UnboundLocalError risk(s) in %s",
                    n_fixes, fname,
                )
    if _total_ub_fixes:
        logger.info(
            "Stage 10: auto-fixed %d total UnboundLocalError risks", _total_ub_fixes
        )

    complexity_warnings: list[str] = []
    for fname, code in files.items():
        if fname.endswith(".py"):
            cw = check_code_complexity(code)
            for w in cw:
                complexity_warnings.append(f"[{fname}] {w}")
                logger.warning("Stage 10 code quality: [%s] %s", fname, w)

    # --- P1.1+P1.2: Deep quality analysis (class quality, scoping, API) ---
    deep_warnings = deep_validate_files(files)
    for w in deep_warnings:
        logger.warning("Stage 10 deep quality: %s", w)
    complexity_warnings.extend(deep_warnings)

    # --- P1.2: If critical deep issues found, attempt one repair cycle ---
    critical_deep = [w for w in deep_warnings if any(
        kw in w for kw in ("UnboundLocalError", "unregistered", "does not exist",
                           "empty or trivial subclass", "does NOT override",
                           "Import-usage mismatch", "NameError",
                           "was removed", "ptp()",
                           "copy-paste", "identical method signatures",
                           "identical AST", "NOT a real ablation",
                           "shadows stdlib/pip")
    )]
    if critical_deep and llm is not None:
        logger.info(
            "Stage 10: %d critical code issues found — triggering repair cycle",
            len(critical_deep),
        )
        repair_issues = "\n".join(f"- {w}" for w in critical_deep)
        all_code_ctx = "\n\n".join(
            f"```filename:{f}\n{c}\n```" for f, c in files.items()
        )
        repair_prompt = (
            f"CRITICAL CODE QUALITY ISSUES FOUND:\n{repair_issues}\n\n"
            f"Fix ALL these issues in the code below. Return the complete "
            f"corrected files using ```filename:xxx.py format.\n\n"
            f"RULES:\n"
            f"- nn.Linear/nn.Conv must be created in __init__(), not forward()\n"
            f"- Variables used after if/else must be defined before the branch\n"
            f"- Use scipy.special.erf, not np.erf\n"
            f"- Ablation/variant classes must have genuinely different logic\n"
            f"- Every class must have a real implementation, not just `pass`\n"
            f"- Ablation classes MUST override the parent method that implements "
            f"the component being ablated (e.g., if ablating attention, override "
            f"the attention method with a simpler alternative like mean pooling)\n"
            f"- IMPORT CONSISTENCY: if you write `from X import Y`, call `Y()` "
            f"directly — NOT `X.Y()`. Mixing styles causes NameError.\n"
            f"- NumPy 2.0: ndarray.ptp() was removed — use arr.max()-arr.min()\n"
            f"- NumPy 2.0: np.bool/np.int/np.float removed — use builtins\n"
            f"- Pretrained models (EfficientNet, ResNet, ViT) expect 224×224 input "
            f"— add `transforms.Resize(224)` when using CIFAR (32×32) or similar\n"
            f"- Copy-paste ablation: if two classes have identical bodies, REWRITE "
            f"the ablation to genuinely remove/reduce a component (e.g., zero out "
            f"attention weights, halve hidden dimensions, remove a loss term)\n"
            f"- KD: teacher must be frozen, add projection layers if teacher_dim != "
            f"student_dim, use temperature T=4 for soft targets\n"
            f"- FILENAME COLLISIONS: If a file like config.py shadows a pip/stdlib "
            f"package, rename it (e.g., config.py → experiment_config.py) and update "
            f"ALL imports referencing it\n\n"
            f"Current code:\n{all_code_ctx}\n"
        )
        try:
            repair_resp = _chat_with_prompt(
                llm,
                _pm.system("code_generation"),
                repair_prompt,
                max_tokens=_code_max_tokens,
            )
            repaired = _extract_multi_file_blocks(repair_resp.content)
            if repaired and "main.py" in repaired:
                files = repaired
                for fname, code in files.items():
                    (exp_dir / fname).write_text(code, encoding="utf-8")
                # Re-check after repair
                deep_warnings_after = deep_validate_files(files)
                fixed = len(critical_deep) - len([
                    w for w in deep_warnings_after
                    if any(kw in w for kw in (
                        "UnboundLocalError", "unregistered", "does not exist",
                        "empty or trivial subclass", "does NOT override",
                        "Import-usage mismatch", "NameError",
                        "was removed", "ptp()",
                        "copy-paste", "identical method signatures",
                        "identical AST", "NOT a real ablation",
                        "shadows stdlib/pip",
                    ))
                ])
                logger.info(
                    "Stage 10: Deep repair fixed %d/%d critical issues",
                    fixed, len(critical_deep),
                )
                complexity_warnings.append(
                    f"[REPAIR] Deep repair fixed {fixed}/{len(critical_deep)} "
                    f"critical issues"
                )
        except Exception as exc:
            logger.debug("Deep repair failed: %s", exc)

    if complexity_warnings:
        health: dict[str, Any] = {}
        health["code_complexity_warnings"] = complexity_warnings
        (stage_dir / "code_complexity.json").write_text(
            json.dumps(health, indent=2), encoding="utf-8"
        )

    # --- P1.4: LLM Code Review (Stage 10.5) ---
    # Skip when CodeAgent is active — Phase 4 review already covers this.
    if llm is not None and not _code_agent_active:
        all_code_review = "\n\n".join(
            f"# --- {fname} ---\n{code}" for fname, code in files.items()
        )
        if len(all_code_review) > 12000:
            all_code_review = all_code_review[:12000] + "\n... [truncated]"
        review_prompt = (
            f"You are a senior researcher reviewing experiment code for a "
            f"research submission.\n\n"
            f"TOPIC: {_authoritative_topic}\n"
            f"EXPERIMENT PLAN:\n{exp_plan[:3000]}\n\n"
            f"CODE:\n```python\n{all_code_review}\n```\n\n"
            f"Review the code and return JSON with this EXACT structure:\n"
            f'{{"score": <1-10>, "issues": ['
            f'{{"severity": "critical|major|minor", '
            f'"description": "...", "fix": "..."}}], '
            f'"verdict": "pass|needs_fix"}}\n\n'
            f"Check specifically:\n"
            f"1. Does each algorithm/method have a DISTINCT implementation? "
            f"(Not just renamed copies)\n"
            f"2. Are ablation conditions genuinely different from the main method?\n"
            f"3. Are loss functions / training loops mathematically correct?\n"
            f"4. Will the code actually run without errors? Check variable scoping, "
            f"API usage, tensor shape compatibility.\n"
            f"5. Is the code complex enough for a research paper? (Not trivial)\n"
            f"6. Are experimental conditions fairly compared (same seeds, data)?\n"
            f"7. If using pretrained models (EfficientNet, ResNet, ViT), are input "
            f"images resized to the model's expected size (e.g., 224x224)? CIFAR "
            f"images are 32x32 and MUST be resized for pretrained models.\n"
            f"8. Are imports consistent? `from X import Y` must use `Y()`, not `X.Y()`.\n"
        )
        try:
            review_resp = llm.chat(
                [{"role": "user", "content": review_prompt}],
                system="You are a meticulous ML code reviewer. Be strict.",
                max_tokens=2048,
            )
            # Extract JSON from LLM response (may be wrapped in markdown fences)
            _review_text = review_resp.content if hasattr(review_resp, "content") else str(review_resp)
            # Strip markdown JSON fences if present
            _review_text = _review_text.strip()
            if _review_text.startswith("```"):
                _lines = _review_text.splitlines()
                _start = 1 if _lines[0].strip().startswith("```") else 0
                _end = len(_lines) - 1 if _lines[-1].strip() == "```" else len(_lines)
                _review_text = "\n".join(_lines[_start:_end])
            review_data = _safe_json_loads(_review_text, {})
            if isinstance(review_data, dict):
                review_score = review_data.get("score", 0)
                review_verdict = review_data.get("verdict", "unknown")
                review_issues = review_data.get("issues", [])

                # Write review report
                review_report = {
                    "score": review_score,
                    "verdict": review_verdict,
                    "issues": review_issues,
                    "timestamp": _utcnow_iso(),
                }
                (stage_dir / "code_review.json").write_text(
                    json.dumps(review_report, indent=2), encoding="utf-8"
                )

                # If critical issues found and score low, attempt fix
                critical_issues = [
                    i for i in review_issues
                    if isinstance(i, dict)
                    and i.get("severity") == "critical"
                ]
                if critical_issues and review_score <= 4:
                    logger.warning(
                        "Stage 10 code review: score=%d, %d critical issues — "
                        "attempting fix",
                        review_score, len(critical_issues),
                    )
                    fix_descriptions = "\n".join(
                        f"- [{i.get('severity', '?')}] {i.get('description', '?')}: "
                        f"{i.get('fix', 'no fix suggested')}"
                        for i in critical_issues
                    )
                    fix_prompt = (
                        f"Code review found {len(critical_issues)} CRITICAL issues "
                        f"(score: {review_score}/10):\n{fix_descriptions}\n\n"
                        f"Fix ALL critical issues. Return complete corrected files "
                        f"using ```filename:xxx.py format.\n\n"
                        f"Current code:\n"
                        + "\n\n".join(
                            f"```filename:{f}\n{c}\n```" for f, c in files.items()
                        )
                    )
                    try:
                        fix_resp = _chat_with_prompt(
                            llm,
                            _pm.system("code_generation"),
                            fix_prompt,
                            max_tokens=_code_max_tokens,
                        )
                        fixed_files = _extract_multi_file_blocks(fix_resp.content)
                        if fixed_files and "main.py" in fixed_files:
                            files = fixed_files
                            for fname, code in files.items():
                                (exp_dir / fname).write_text(code, encoding="utf-8")
                            logger.info(
                                "Stage 10: Code fixed after review "
                                "(was %d/10, %d critical issues)",
                                review_score, len(critical_issues),
                            )
                    except Exception as exc:
                        logger.debug("Review-fix failed: %s", exc)
        except Exception as exc:
            logger.debug("Code review failed: %s", exc)

    # --- FIX-3: Topic-experiment alignment check ---
    # BUG-171: Previous 8000-char truncation caused false-positive misalignment
    # for multi-file experiments (30-90K chars). LLM saw "[truncated]" and
    # concluded code was incomplete. Fix: build a structured summary that
    # includes file inventory + full main.py + per-file function/class headers.
    alignment_ok = True
    alignment_note = ""
    if llm is not None:
        # Build structured code summary for alignment check
        _file_inventory = []
        for _fn, _cd in files.items():
            _lines = _cd.count("\n") + 1
            _file_inventory.append(f"  {_fn}: {_lines} lines, {len(_cd)} chars")
        _inventory_block = "FILES GENERATED:\n" + "\n".join(_file_inventory)

        # BUG-179: Beast Mode may use a different entry point (e.g.
        # run_experiment.py).  Detect the actual entry point by scanning
        # for ``if __name__ == "__main__"`` in all files, preferring main.py.
        _entry_file = "main.py"
        if "main.py" not in files or not files.get("main.py", "").strip():
            for _fn, _cd in files.items():
                if 'if __name__' in _cd and '__main__' in _cd:
                    _entry_file = _fn
                    break
        elif files.get("main.py", ""):
            # main.py exists but may be a stub — if another file has the
            # real orchestration (more lines + __main__ guard), prefer it
            _main_lines = files["main.py"].count("\n")
            for _fn, _cd in files.items():
                if _fn == "main.py":
                    continue
                if ('if __name__' in _cd and '__main__' in _cd
                        and _cd.count("\n") > _main_lines * 1.5):
                    _entry_file = _fn
                    break

        _main_code = files.get(_entry_file, files.get("main.py", ""))
        _main_block = f"# --- {_entry_file} (FULL — entry point) ---\n{_main_code}"
        # Cap main.py at 12000 chars to stay within token budget
        if len(_main_block) > 12000:
            _main_block = _main_block[:12000] + "\n... [main.py truncated at 12000 chars]"

        # For other files, include imports + function/class signatures
        _other_summaries = []
        for _fn, _cd in files.items():
            if _fn == _entry_file:
                continue
            _sig_lines = []
            for _line in _cd.split("\n"):
                _stripped = _line.strip()
                if (_stripped.startswith("def ") or _stripped.startswith("class ")
                        or _stripped.startswith("async def ")
                        # BUG-209: Include import lines — they reveal which
                        # techniques/libraries are used (e.g. CosineAnnealingLR)
                        or _stripped.startswith("import ")
                        or _stripped.startswith("from ")):
                    _sig_lines.append(_line)
            if _sig_lines:
                _other_summaries.append(
                    f"# --- {_fn} (imports + signatures) ---\n"
                    + "\n".join(_sig_lines)
                )
            else:
                # Small file — include first 800 chars
                _preview = _cd[:800]
                if len(_cd) > 800:
                    _preview += f"\n... [{len(_cd) - 800} more chars]"
                _other_summaries.append(f"# --- {_fn} (preview) ---\n{_preview}")
        _other_block = "\n\n".join(_other_summaries)
        # Cap other summaries
        if len(_other_block) > 6000:
            _other_block = _other_block[:6000] + "\n... [other files truncated]"

        all_code_for_check = (
            f"{_inventory_block}\n\n{_main_block}\n\n{_other_block}"
        )
        align_prompt = (
            f"Research topic: {_authoritative_topic}\n\n"
            f"Experiment code:\n```python\n{all_code_for_check}\n```\n\n"
            "TASK: Evaluate whether this experiment code actually tests the "
            "stated research topic. Answer with JSON:\n"
            '{"aligned": true/false, "reason": "...", "suggestions": "..."}\n\n'
            "IMPORTANT: The code spans MULTIPLE files. The file inventory above "
            "shows ALL generated files. Only main.py is shown in full; other "
            "files show function/class signatures. Do NOT mark as misaligned "
            "just because helper files are summarized — they contain full "
            "implementations.\n\n"
            "Check specifically:\n"
            "- Does main.py orchestrate an experiment matching the topic?\n"
            "- Do the helper file signatures indicate relevant models/methods?\n"
            "- If the topic mentions a specific technique, is there evidence of "
            "its implementation (function names, class names, imports)?\n"
            "- Are the experimental conditions meaningfully different from each other?\n"
        )
        try:
            align_resp = llm.chat(
                [{"role": "user", "content": align_prompt}],
                system="You are a scientific code reviewer checking topic-experiment alignment.",
                max_tokens=1024,
            )
            align_data = _safe_json_loads(align_resp.content, {})
            if isinstance(align_data, dict) and not align_data.get("aligned", True):
                alignment_ok = False
                alignment_note = align_data.get("reason", "Misaligned")
                suggestions = align_data.get("suggestions", "")
                logger.warning(
                    "Stage 10: Topic-experiment MISALIGNMENT detected: %s",
                    alignment_note,
                )
                # BUG-R6-01: Allow up to 2 regeneration attempts with re-check.
                _max_regen = 2
                for _regen_attempt in range(1, _max_regen + 1):
                    logger.info(
                        "Stage 10: Alignment regen attempt %d/%d",
                        _regen_attempt, _max_regen,
                    )
                    regen_prompt = (
                        f"The experiment code you previously generated does NOT align "
                        f"with the research topic.\n\n"
                        f"TOPIC: {_authoritative_topic}\n"
                        f"MISALIGNMENT: {alignment_note}\n"
                        f"SUGGESTIONS: {suggestions}\n\n"
                        f"REGENERATE the experiment code to DIRECTLY test the stated "
                        f"topic. The code MUST implement the core technique described "
                        f"in the topic, not a generic proxy.\n\n"
                        f"CRITICAL CONSTRAINTS:\n"
                        f"- You MUST implement the EXACT approved manifest below.\n"
                        f"- Do NOT substitute any model, benchmark, method, or "
                        f"resource count.\n"
                        f"{_manifest_guidance}\n\n"
                        f"{pkg_hint}\n{compute_budget}\n"
                        f"PLAN:\n{exp_plan}\n\n"
                        f"Return multiple files using ```filename:xxx.py format."
                    )
                    regen_resp = _chat_with_prompt(
                        llm,
                        system=_pm.system("code_generation"),
                        user=regen_prompt,
                        max_tokens=_code_max_tokens,
                    )
                    regen_files = _extract_multi_file_blocks(regen_resp.content)
                    if not regen_files or "main.py" not in regen_files:
                        logger.warning(
                            "Stage 10: Regen attempt %d produced no main.py",
                            _regen_attempt,
                        )
                        continue
                    files = regen_files
                    for fname, code in files.items():
                        (exp_dir / fname).write_text(code, encoding="utf-8")
                    # Re-check alignment on regenerated code (BUG-171 fix)
                    _rc_inv = []
                    for _fn, _cd in files.items():
                        _rc_inv.append(f"  {_fn}: {_cd.count(chr(10))+1} lines")
                    _rc_main = files.get("main.py", "")
                    if len(_rc_main) > 12000:
                        _rc_main = _rc_main[:12000] + "\n... [truncated]"
                    _rc_sigs = []
                    for _fn, _cd in files.items():
                        if _fn == "main.py":
                            continue
                        # BUG-209: Include imports alongside signatures
                        _slines = [l for l in _cd.split("\n")
                                   if l.strip().startswith((
                                       "def ", "class ", "async def ",
                                       "import ", "from ",
                                   ))]
                        if _slines:
                            _rc_sigs.append(f"# {_fn} imports+signatures:\n" + "\n".join(_slines))
                    recheck_code = (
                        "FILES:\n" + "\n".join(_rc_inv) + "\n\n"
                        f"# main.py (FULL):\n{_rc_main}\n\n"
                        + "\n".join(_rc_sigs)
                    )
                    recheck_resp = llm.chat(
                        [{"role": "user", "content": (
                            f"Research topic: {_authoritative_topic}\n\n"
                            f"Experiment code:\n```python\n{recheck_code}\n```\n\n"
                            "TASK: Evaluate whether this experiment code actually tests "
                            "the stated research topic. Only main.py is shown in full; "
                            "other files show signatures only. Answer with JSON:\n"
                            '{"aligned": true/false, "reason": "...", "suggestions": "..."}\n'
                        )}],
                        system="You are a scientific code reviewer checking topic-experiment alignment.",
                        max_tokens=1024,
                    )
                    recheck_data = _safe_json_loads(recheck_resp.content, {})
                    if isinstance(recheck_data, dict) and recheck_data.get("aligned", False):
                        alignment_ok = True
                        alignment_note = f"Regenerated after alignment check (attempt {_regen_attempt})"
                        logger.info(
                            "Stage 10: Code aligned after regen attempt %d",
                            _regen_attempt,
                        )
                        break
                    else:
                        alignment_note = recheck_data.get("reason", alignment_note)
                        suggestions = recheck_data.get("suggestions", suggestions)
                        logger.warning(
                            "Stage 10: Regen attempt %d still misaligned: %s",
                            _regen_attempt, alignment_note,
                        )
        except Exception as exc:
            logger.debug("Alignment check failed: %s", exc)

    # --- FIX-7: Ablation distinctness check ---
    main_code = files.get("main.py", "")
    if llm is not None and main_code and "condition" in main_code.lower():
        try:
            ablation_context = _ablation_review_context(files)
            ablation_prompt = (
                f"Examine this multi-file experiment code:\n"
                f"```python\n{ablation_context}\n```\n\n"
                "Check if any experimental conditions (methods/ablations) have "
                "IDENTICAL configurations (same hyperparameters, same code paths). "
                "Only report duplicates when the implementation is visible and "
                "you can name the concrete duplicate pair and code evidence. "
                "If evidence is incomplete or uncertain, set has_duplicates=false. "
                "Answer JSON: "
                '{"has_duplicates": true/false, "duplicate_pairs": ['
                '{"left": "condition_a", "right": "condition_b", '
                '"evidence": "specific identical implementation"}], '
                '"details": "short explanation"}'
            )
            abl_resp = llm.chat(
                [{"role": "user", "content": ablation_prompt}],
                system="You are a code reviewer checking experimental conditions.",
                max_tokens=512,
            )
            abl_data = _safe_json_loads(abl_resp.content, {})
            if _confirmed_ablation_duplicates(abl_data):
                logger.warning(
                    "Stage 10: Duplicate ablation conditions detected: %s",
                    abl_data.get("details", ""),
                )
                (stage_dir / "ablation_warning.json").write_text(
                    json.dumps(abl_data, indent=2), encoding="utf-8"
                )
                # --- Attempt ablation repair ---
                all_code_ctx = "\n\n".join(
                    f"```filename:{f}\n{c}\n```" for f, c in files.items()
                )
                dup_details = abl_data.get("details", "unknown")
                abl_repair_prompt = (
                    f"ABLATION REPAIR REQUIRED — duplicate conditions detected:\n"
                    f"{dup_details}\n\n"
                    f"Rewrite the ablation/variant conditions so each one is "
                    f"GENUINELY DIFFERENT. Concrete strategies:\n"
                    f"- 'no_<component>': REMOVE the component entirely "
                    f"(e.g., replace attention with mean pooling, remove a loss term)\n"
                    f"- 'reduced_capacity': HALVE hidden dimensions or layers\n"
                    f"- Different conditions MUST produce different outputs on the "
                    f"same input. Add a startup assertion that runs one forward pass "
                    f"per condition on identical input and prints:\n"
                    f"  ABLATION_CHECK: <cond1> vs <cond2> outputs_differ=True\n\n"
                    f"Return ALL files using ```filename:xxx.py format.\n\n"
                    f"Current code:\n{all_code_ctx}\n"
                )
                try:
                    abl_repair_resp = _chat_with_prompt(
                        llm,
                        _pm.system("code_generation"),
                        abl_repair_prompt,
                        max_tokens=_code_max_tokens,
                    )
                    repaired_files = _extract_multi_file_blocks(
                        abl_repair_resp.content
                    )
                    if repaired_files and "main.py" in repaired_files:
                        files = repaired_files
                        for fname, code in files.items():
                            (exp_dir / fname).write_text(code, encoding="utf-8")
                        logger.info(
                            "Stage 10: Ablation repair applied — "
                            "rewrote duplicate conditions"
                        )
                except Exception as exc:
                    logger.debug("Ablation repair failed: %s", exc)
        except Exception as exc:
            logger.debug("Ablation validation skipped: %s", exc)

    # --- Write spec ---
    scientific_report = _assess_scientific_code_alignment(
        files,
        _load_scientific_contract(run_dir, config),
        _authoritative_topic,
    )
    scientific_gate = _scientific_code_alignment_result(
        stage_dir,
        scientific_report,
    )
    pilot_report = _assess_pilot_envelope(
        files,
        _load_scientific_contract(run_dir, config),
    )
    (stage_dir / "pilot_envelope.json").write_text(
        json.dumps(pilot_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    file_list = ", ".join(f"`{f}`" for f in sorted(files.keys()))
    main_validation = validate_code(files.get("main.py", ""))
    _align_status = "ALIGNED" if alignment_ok else f"MISALIGNED: {alignment_note}"
    spec = f"""# Experiment Specification

## Topic
{_authoritative_topic}

## Project Structure
Multi-file experiment project with {len(files)} file(s): {file_list}

## Entry Point
`main.py` \u2014 executed directly via sandbox

## Outputs
- `main.py` emits metric lines in `name: value` format
- Primary metric key: `{metric}`

## Topic-Experiment Alignment
{_align_status}

## Constraints
- Time budget per run: {config.experiment.time_budget_sec}s
- Max iterations: {config.experiment.max_iterations}
- Self-contained execution (no external data, no network)
- Validated: {main_validation.summary()}

## Generated
{_utcnow_iso()}
"""
    (stage_dir / "experiment_spec.md").write_text(spec, encoding="utf-8")

    artifacts = [
        "experiment/",
        "experiment_spec.md",
        "scientific_code_alignment.json",
        "pilot_envelope.json",
    ]
    if (stage_dir / "implementation_contract.json").exists():
        artifacts.extend(
            [
                "implementation_contract.json",
                "implementation_manifest.json",
                "implementation_manifest_gate.json",
            ]
        )
    if (stage_dir / "validation_report.md").exists():
        artifacts.append("validation_report.md")

    if scientific_gate is not None:
        return StageResult(
            stage=scientific_gate.stage,
            status=scientific_gate.status,
            artifacts=tuple(artifacts),
            error=scientific_gate.error,
            decision=scientific_gate.decision,
            evidence_refs=scientific_gate.evidence_refs,
        )
    if not pilot_report.get("aligned", False):
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.PAUSED,
            artifacts=tuple(artifacts),
            error=(
                "Generated code expands the selected topic's cheap pilot "
                "beyond the production envelope"
            ),
            decision="pilot_envelope_violation",
            evidence_refs=("stage-10/pilot_envelope.json",),
        )

    # BUG-R6-01: Fail stage if alignment check detected persistent mismatch
    # after all regen attempts, instead of silently proceeding.
    if not alignment_ok:
        logger.error(
            "Stage 10: Persistent topic-experiment misalignment after all "
            "regen attempts. Failing stage. Reason: %s",
            alignment_note,
        )
        return StageResult(
            stage=Stage.CODE_GENERATION,
            status=StageStatus.FAILED,
            artifacts=tuple(artifacts),
            evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
            error=f"Topic-experiment misalignment: {alignment_note}",
        )

    return StageResult(
        stage=Stage.CODE_GENERATION,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-10/{a}" for a in artifacts),
    )
