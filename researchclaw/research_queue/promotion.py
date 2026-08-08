"""Idea-to-treatment promotion bridge and benchmark outcome separation."""

from __future__ import annotations

import ast
import json
import shutil
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from researchclaw.benchmark_adapter.cifar10_calibration import sha256_path

from .benchmark_profile import (
    BenchmarkCompatibility,
    build_benchmark_plan,
    load_benchmark_profile,
    validate_benchmark_compatibility,
)
from .benchmark_runner import (
    benchmark_command,
    build_promoted_benchmark_config,
)
from .config import BenchmarkPromotionConfig
from .models import (
    BudgetLevel,
    IdeaRecord,
    PreparedRevision,
    ResearchSpec,
    RunRecord,
    RunStatus,
    new_id,
    utc_now,
)
from .scientific_gate import (
    hypothesis_supported,
    validate_benchmark_result,
)
from .store import ResearchQueueStore
from .treatment_preflight import preflight_treatment
from .workers import TreatmentWorker, validate_python_sources


@dataclass(frozen=True, slots=True)
class PromotionOutcome:
    execution_passed: bool
    scientific_valid: bool
    hypothesis_supported: bool | None
    promotion_decision: str
    reason: str
    benchmark_result: dict[str, Any]
    scientific_gate: dict[str, Any]
    usage: dict[str, Any]
    provenance: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_passed": self.execution_passed,
            "scientific_valid": self.scientific_valid,
            "hypothesis_supported": self.hypothesis_supported,
            "promotion_decision": self.promotion_decision,
            "reason": self.reason,
            "benchmark_result": self.benchmark_result,
            "scientific_gate": self.scientific_gate,
            "usage": self.usage,
            "provenance": self.provenance,
        }


@dataclass(frozen=True, slots=True)
class ProgressivePreparationOutcome:
    """Result of generating one immutable treatment-backed Queue revision."""

    passed: bool
    reason: str
    prepared_revision: PreparedRevision | None
    usage: dict[str, Any]
    preflight: dict[str, Any]


def review_benchmark_result(
    *,
    spec: ResearchSpec,
    benchmark_result: Mapping[str, Any],
    execution_passed: bool,
    minimum_effect: float = 0.0,
    execution_error: str = "",
    usage: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> PromotionOutcome:
    """Apply the deterministic scientific contract to an existing result."""

    scientific = validate_benchmark_result(spec, benchmark_result)
    scientific_valid = bool(execution_passed and scientific.passed)
    supported = (
        hypothesis_supported(
            spec,
            benchmark_result,
            minimum_effect=minimum_effect,
        )
        if scientific_valid
        else None
    )
    decision = "accept" if supported is True else "reject"
    if decision == "accept":
        reason = (
            "treatment improved the primary metric and passed all "
            "machine-executable evidence requirements"
        )
    elif scientific.errors:
        reason = "; ".join(scientific.errors)
    else:
        reason = execution_error or "hypothesis was not supported on the real benchmark"
    return PromotionOutcome(
        execution_passed=bool(execution_passed),
        scientific_valid=scientific_valid,
        hypothesis_supported=supported,
        promotion_decision=decision,
        reason=reason,
        benchmark_result=dict(benchmark_result),
        scientific_gate=scientific.to_dict(),
        usage=dict(usage or {}),
        provenance=dict(provenance or {}),
    )


def re_review_artifacts(
    *,
    idea_dir: str | Path,
    minimum_effect: float = 0.0,
) -> PromotionOutcome:
    """Re-review persisted evidence without rerunning code, LLMs, or GPUs."""

    root = Path(idea_dir).expanduser().resolve()
    benchmark_root = root / "benchmark"
    spec = ResearchSpec.from_mapping(_read_json(benchmark_root / "research_spec.json"))
    benchmark_result = _read_json(benchmark_root / "result.json")
    previous = _read_json(root / "final_review.json")
    if not previous:
        previous = _read_json(benchmark_root / "final_review.json")
    outcome = review_benchmark_result(
        spec=spec,
        benchmark_result=benchmark_result,
        execution_passed=str(benchmark_result.get("status", "")).casefold()
        in {"ok", "success", "succeeded", "passed"},
        minimum_effect=minimum_effect,
        execution_error=str(benchmark_result.get("error", "") or ""),
        usage=previous.get("usage", {}),
        provenance=previous.get("provenance", {}),
    )
    for path in (
        benchmark_root / "final_review.json",
        root / "final_review.json",
    ):
        _write_json_atomic(path, outcome.to_dict())
    return outcome


class BenchmarkPromotionBridge:
    def __init__(
        self,
        *,
        config: BenchmarkPromotionConfig,
        store: ResearchQueueStore,
        treatment_worker: TreatmentWorker,
        run_backend: Any,
        max_gpus_per_run: int,
    ) -> None:
        self.config = config
        self.store = store
        self.treatment_worker = treatment_worker
        self.run_backend = run_backend
        self.max_gpus_per_run = max(1, int(max_gpus_per_run))
        self.profile = load_benchmark_profile(
            config.benchmark_id,
            config.benchmark_config,
        )
        self.benchmark_plan = build_benchmark_plan(
            profile=self.profile,
            config_path=config.benchmark_config,
        )

    def compatibility(self, spec: ResearchSpec) -> BenchmarkCompatibility:
        return validate_benchmark_compatibility(spec, self.profile)

    def profile_dict(self) -> dict[str, Any]:
        return self.profile.to_dict()

    def _persist_contract(
        self,
        idea: IdeaRecord,
        spec: ResearchSpec,
    ) -> tuple[Path, BenchmarkCompatibility]:
        benchmark_root = self.store.idea_dir(idea.idea_id) / "benchmark"
        benchmark_root.mkdir(parents=True, exist_ok=True)
        self.store.write_json_atomic(
            benchmark_root / "research_spec.json",
            spec.to_dict(),
        )
        self.store.write_json_atomic(
            benchmark_root / "benchmark_plan.json",
            self.benchmark_plan,
        )
        compatibility = self.compatibility(spec)
        self.store.write_json_atomic(
            benchmark_root / "benchmark_compatibility.json",
            compatibility.to_dict(),
        )
        return benchmark_root, compatibility

    def _prepare_treatment(
        self,
        idea: IdeaRecord,
        *,
        spec: ResearchSpec,
        benchmark_root: Path,
    ) -> tuple[Path | None, dict[str, Any], dict[str, Any]]:
        """Generate and preflight one treatment, reusing it after a resume."""

        treatment_path = benchmark_root / "treatment.py"
        treatment_manifest = _read_json(
            benchmark_root / "treatment-manifest.json"
        )
        current_spec_sha256 = sha256_path(
            benchmark_root / "research_spec.json"
        )
        if (
            treatment_path.is_file()
            and treatment_manifest.get("research_spec_sha256")
            == current_spec_sha256
        ):
            source_errors = validate_treatment_source(
                treatment_path.read_text(encoding="utf-8")
            )
            preflight = (
                {
                    "passed": False,
                    "errors": source_errors,
                    "reused": True,
                }
                if source_errors
                else {
                    **preflight_treatment(
                        treatment_path,
                        examples=self.config.preflight_examples,
                        classes=self.config.preflight_classes,
                        timeout_sec=self.config.preflight_timeout_sec,
                    ),
                    "reused": True,
                }
            )
            self.store.write_json_atomic(
                benchmark_root / "preflight-reuse.json",
                preflight,
            )
            if preflight.get("passed"):
                return treatment_path, {}, preflight

        usage: dict[str, Any] = {}
        feedback = ""
        preflight: dict[str, Any] = {}
        for attempt in range(self.config.max_treatment_repairs + 1):
            try:
                source, current_usage = self.treatment_worker.build(
                    idea,
                    spec=spec,
                    feedback=feedback,
                )
            except Exception as exc:  # noqa: BLE001
                preflight = {
                    "passed": False,
                    "errors": [
                        (
                            "treatment generation failed validation: "
                            f"{type(exc).__name__}: {exc}"
                        )
                    ],
                }
                self.store.write_json_atomic(
                    benchmark_root / f"preflight-{attempt + 1:02d}.json",
                    preflight,
                )
                # StructuredRole already performs bounded repair internally.
                # There is no metered usable result object to retry here.
                break
            _merge_usage(usage, current_usage)
            source_errors = validate_treatment_source(source)
            treatment_path.write_text(source, encoding="utf-8")
            if source_errors:
                preflight = {
                    "passed": False,
                    "errors": source_errors,
                }
            else:
                preflight = preflight_treatment(
                    treatment_path,
                    examples=self.config.preflight_examples,
                    classes=self.config.preflight_classes,
                    timeout_sec=self.config.preflight_timeout_sec,
                )
            self.store.write_json_atomic(
                benchmark_root / f"preflight-{attempt + 1:02d}.json",
                preflight,
            )
            if preflight.get("passed"):
                break
            feedback = (
                "Treatment preflight failed. Return a complete corrected "
                f"treatment. Diagnostics: {json.dumps(preflight, ensure_ascii=False)}"
            )
        if not preflight.get("passed"):
            return None, usage, preflight
        self.store.write_json_atomic(
            benchmark_root / "treatment-manifest.json",
            {
                "treatment_sha256": sha256_path(treatment_path),
                "research_spec_sha256": current_spec_sha256,
                "preflight": preflight,
                "usage": usage,
            },
        )
        return treatment_path, usage, preflight

    def prepare_progressive_revision(
        self,
        idea: IdeaRecord,
        *,
        spec: ResearchSpec,
        revision: int,
        timeout_sec: float,
    ) -> ProgressivePreparationOutcome:
        """Create one framework-owned runner around one generated treatment."""

        benchmark_root, compatibility = self._persist_contract(idea, spec)
        if not compatibility.passed:
            return ProgressivePreparationOutcome(
                passed=False,
                reason=(
                    "ResearchSpec is incompatible with the frozen benchmark "
                    "profile: " + "; ".join(compatibility.errors)
                ),
                prepared_revision=None,
                usage={},
                preflight={
                    "passed": False,
                    "errors": list(compatibility.errors),
                },
            )
        treatment_path, usage, preflight = self._prepare_treatment(
            idea,
            spec=spec,
            benchmark_root=benchmark_root,
        )
        if treatment_path is None:
            return ProgressivePreparationOutcome(
                passed=False,
                reason="generated treatment failed deterministic preflight",
                prepared_revision=None,
                usage=usage,
                preflight=preflight,
            )
        logits_cache = (
            Path(self.config.logits_cache).expanduser().resolve()
            if self.config.logits_cache
            else None
        )
        cache_ready = bool(
            self.config.prefer_logits_cache
            and logits_cache is not None
            and logits_cache.is_file()
        )
        command = [
            sys.executable if cache_ready else self.config.runtime_python,
            "-m",
            "researchclaw.research_queue.progressive_benchmark_runner",
            "--benchmark-config",
            str(Path(self.config.benchmark_config).expanduser().resolve()),
            "--treatment-path",
            str(treatment_path),
        ]
        if cache_ready and logits_cache is not None:
            command += ["--logits-cache", str(logits_cache)]
        prepared = PreparedRevision(
            revision=revision,
            command=tuple(command),
            requested_gpus=0 if cache_ready else 1,
            timeout_sec=max(1.0, float(timeout_sec)),
            plan={
                "method_summary": (
                    "One immutable generated treatment evaluated by the "
                    "framework-owned progressive benchmark runner."
                ),
                "treatment": spec.treatment,
                "control": spec.control,
                "primary_metric": spec.primary_metric,
                "source_files": {},
                "benchmark_config": str(
                    Path(self.config.benchmark_config).expanduser().resolve()
                ),
                "benchmark_config_sha256": sha256_path(
                    self.config.benchmark_config
                ),
                "treatment_path": str(treatment_path),
                "treatment_sha256": sha256_path(treatment_path),
                "logits_cache": str(logits_cache or ""),
                "logits_cache_sha256": (
                    sha256_path(logits_cache)
                    if cache_ready and logits_cache is not None
                    else ""
                ),
                "progressive_evidence": True,
            },
            usage=usage,
        )
        return ProgressivePreparationOutcome(
            passed=True,
            reason="immutable treatment passed deterministic preflight",
            prepared_revision=prepared,
            usage=usage,
            preflight=preflight,
        )

    def finalize_progressive(
        self,
        idea: IdeaRecord,
        *,
        spec: ResearchSpec,
        run: RunRecord,
    ) -> PromotionOutcome:
        """Apply the final scientific gate to the already completed B2 Run."""

        benchmark_root, compatibility = self._persist_contract(idea, spec)
        raw_path = Path(run.output_dir) / "benchmark-result.json"
        benchmark_result = _read_json(raw_path)
        execution_passed = bool(
            compatibility.passed
            and run.status is RunStatus.SUCCEEDED
            and str(benchmark_result.get("status", "")).casefold()
            in {"ok", "success", "succeeded", "passed"}
        )
        logits_cache = (
            Path(self.config.logits_cache).expanduser().resolve()
            if self.config.logits_cache
            else None
        )
        treatment_path = benchmark_root / "treatment.py"
        provenance = {
            "idea_id": idea.idea_id,
            "revision": run.revision,
            "run_id": run.run_id,
            "budget": run.budget.value,
            "research_spec_sha256": sha256_path(
                benchmark_root / "research_spec.json"
            ),
            "treatment_sha256": (
                sha256_path(treatment_path)
                if treatment_path.is_file()
                else ""
            ),
            "benchmark_plan_sha256": sha256_path(
                benchmark_root / "benchmark_plan.json"
            ),
            "benchmark_profile_version": self.profile.version,
            "benchmark_id": self.config.benchmark_id,
            "logits_cache_sha256": (
                sha256_path(logits_cache)
                if logits_cache is not None and logits_cache.is_file()
                else ""
            ),
            "evidence_partition": "B2",
        }
        outcome = review_benchmark_result(
            spec=spec,
            benchmark_result=benchmark_result,
            execution_passed=execution_passed,
            minimum_effect=max(
                self.config.minimum_effect,
                self.profile.minimum_effect_for(spec.primary_metric),
            ),
            execution_error=run.error,
            usage={"benchmark": dict(run.result.get("usage", {}) or {})},
            provenance=provenance,
        )
        self.store.write_json_atomic(
            benchmark_root / "final_review.json",
            outcome.to_dict(),
        )
        if raw_path.is_file():
            shutil.copy2(raw_path, benchmark_root / "result.json")
        self.store.event(
            "benchmark_completed",
            idea_id=idea.idea_id,
            run_id=run.run_id,
            execution_passed=outcome.execution_passed,
            scientific_valid=outcome.scientific_valid,
            hypothesis_supported=outcome.hypothesis_supported,
            promotion_decision=outcome.promotion_decision,
            reason=outcome.reason,
            reused_progressive_b2=True,
        )
        return outcome

    async def promote(
        self,
        idea: IdeaRecord,
        *,
        spec: ResearchSpec,
    ) -> PromotionOutcome:
        benchmark_root, compatibility = self._persist_contract(idea, spec)
        if not compatibility.passed:
            outcome = PromotionOutcome(
                execution_passed=False,
                scientific_valid=False,
                hypothesis_supported=None,
                promotion_decision="reject",
                reason=(
                    "ResearchSpec is incompatible with the frozen benchmark "
                    "profile: " + "; ".join(compatibility.errors)
                ),
                benchmark_result={},
                scientific_gate={
                    "passed": False,
                    "errors": list(compatibility.errors),
                    "checks": dict(compatibility.checks),
                },
                usage={},
                provenance={
                    "idea_id": idea.idea_id,
                    "benchmark_id": self.profile.benchmark_id,
                    "benchmark_profile_version": self.profile.version,
                    "research_spec_sha256": sha256_path(
                        benchmark_root / "research_spec.json"
                    ),
                    "benchmark_plan_sha256": sha256_path(
                        benchmark_root / "benchmark_plan.json"
                    ),
                },
            )
            self.store.write_json_atomic(
                benchmark_root / "final_review.json",
                outcome.to_dict(),
            )
            self.store.event(
                "benchmark_compatibility_rejected",
                idea_id=idea.idea_id,
                errors=list(compatibility.errors),
            )
            return outcome
        treatment_path, usage, preflight = self._prepare_treatment(
            idea,
            spec=spec,
            benchmark_root=benchmark_root,
        )
        if treatment_path is None:
            outcome = PromotionOutcome(
                execution_passed=False,
                scientific_valid=False,
                hypothesis_supported=None,
                promotion_decision="reject",
                reason="generated treatment failed deterministic preflight",
                benchmark_result={},
                scientific_gate=preflight,
                usage=usage,
                provenance={
                    "idea_id": idea.idea_id,
                    "research_spec_sha256": sha256_path(
                        benchmark_root / "research_spec.json"
                    ),
                    "treatment_sha256": (
                        sha256_path(benchmark_root / "treatment.py")
                        if (benchmark_root / "treatment.py").is_file()
                        else ""
                    ),
                },
            )
            self.store.write_json_atomic(
                benchmark_root / "final_review.json",
                outcome.to_dict(),
            )
            return outcome

        output_dir = benchmark_root / "output"
        runtime_config = build_promoted_benchmark_config(
            template_path=self.config.benchmark_config,
            treatment_path=treatment_path,
            output_dir=output_dir,
            destination=benchmark_root / "benchmark-config.yaml",
        )
        template = _benchmark_mapping(runtime_config)
        require_cuda = bool(template.get("require_cuda", True))
        device = str(template.get("device", "cuda") or "cuda")
        logits_cache = (
            Path(self.config.logits_cache).expanduser().resolve()
            if self.config.logits_cache
            else None
        )
        cache_ready = bool(
            self.config.prefer_logits_cache
            and logits_cache is not None
            and logits_cache.is_file()
        )
        requested_gpus = (
            0
            if cache_ready
            else (1 if require_cuda or device.startswith("cuda") else 0)
        )
        requested_gpus = min(self.max_gpus_per_run, requested_gpus)
        timeout_sec = max(
            30.0,
            float(template.get("timeout_sec", 1800.0) or 1800.0),
        )
        run_id = new_id("benchmark")
        run = RunRecord(
            run_id=run_id,
            idea_id=idea.idea_id,
            revision=max(1, idea.current_revision),
            budget=BudgetLevel.B2,
            requested_gpus=requested_gpus,
            timeout_sec=timeout_sec,
            command=benchmark_command(
                runtime_config,
                # Zero-GPU cache evaluations run on the controller through
                # LocalRunBackend, not in the node-only torch environment.
                python_executable=(
                    sys.executable
                    if cache_ready
                    else self.config.runtime_python
                ),
                logits_cache=logits_cache if cache_ready else None,
            ),
            output_dir=str(output_dir),
            status=RunStatus.RUNNING,
            started_at=utc_now(),
        )
        self.store.upsert_run(run)
        self.store.event(
            "benchmark_started",
            idea_id=idea.idea_id,
            run_id=run_id,
            requested_gpus=requested_gpus,
            benchmark_id=self.config.benchmark_id,
        )
        result = await self.run_backend.run(
            run,
            revision_dir=benchmark_root,
            output_dir=output_dir,
            env={
                "RESEARCH_QUEUE_IDEA_ID": idea.idea_id,
                "RESEARCH_QUEUE_RUN_ID": run_id,
                "RESEARCH_QUEUE_BENCHMARK_ID": self.config.benchmark_id,
            },
        )
        run.result = result.to_dict()
        run.error = result.error
        run.status = RunStatus.SUCCEEDED if result.ok else RunStatus.FAILED
        run.finished_at = utc_now()
        self.store.upsert_run(run)
        idea.gpu_seconds += float(result.usage.get("gpu_seconds", 0.0) or 0.0)
        benchmark_result = _read_json(output_dir / "result.json")
        if not benchmark_result and result.ok:
            benchmark_result = {
                "status": "ok",
                "metrics": result.metrics,
                "usage": result.usage,
            }
        provenance = {
            "idea_id": idea.idea_id,
            "revision": idea.current_revision,
            "research_spec_sha256": sha256_path(benchmark_root / "research_spec.json"),
            "treatment_sha256": sha256_path(benchmark_root / "treatment.py"),
            "benchmark_config_sha256": sha256_path(runtime_config),
            "benchmark_plan_sha256": sha256_path(
                benchmark_root / "benchmark_plan.json"
            ),
            "benchmark_profile_version": self.profile.version,
            "benchmark_id": self.config.benchmark_id,
            "logits_cache_sha256": (
                sha256_path(logits_cache)
                if cache_ready and logits_cache is not None
                else ""
            ),
        }
        outcome = review_benchmark_result(
            spec=spec,
            benchmark_result=benchmark_result,
            execution_passed=result.ok,
            minimum_effect=max(
                self.config.minimum_effect,
                self.profile.minimum_effect_for(spec.primary_metric),
            ),
            execution_error=result.error,
            usage={
                **usage,
                "benchmark": result.usage,
            },
            provenance=provenance,
        )
        self.store.write_json_atomic(
            benchmark_root / "final_review.json",
            outcome.to_dict(),
        )
        if (output_dir / "result.json").is_file():
            shutil.copy2(output_dir / "result.json", benchmark_root / "result.json")
        self.store.event(
            "benchmark_completed",
            idea_id=idea.idea_id,
            run_id=run_id,
            **{
                key: value
                for key, value in outcome.to_dict().items()
                if key
                in {
                    "execution_passed",
                    "scientific_valid",
                    "hypothesis_supported",
                    "promotion_decision",
                    "reason",
                }
            },
        )
        return outcome


def validate_treatment_source(source: str) -> list[str]:
    errors = validate_python_sources(
        {"treatment.py": source},
        allowed_imports=("numpy",),
    )
    try:
        tree = ast.parse(source, filename="treatment.py")
    except SyntaxError:
        return errors
    definitions = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    if "build_treatment" not in definitions:
        errors.append("treatment.py must define build_treatment()")
    forbidden_names = {
        "eval",
        "exec",
        "open",
        "compile",
        "__import__",
        "getattr",
        "setattr",
        "delattr",
    }
    forbidden_modules = {
        "os",
        "pathlib",
        "subprocess",
        "socket",
        "requests",
        "httpx",
        "urllib",
        "importlib",
        "builtins",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            errors.append(f"treatment.py uses forbidden name {node.id!r}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in forbidden_modules:
                    errors.append(
                        f"treatment.py imports forbidden module {alias.name!r}"
                    )
        if (
            isinstance(node, ast.ImportFrom)
            and node.module
            and node.module.split(".", 1)[0] in forbidden_modules
        ):
            errors.append(f"treatment.py imports forbidden module {node.module!r}")
    lowered = source.casefold()
    if "evaluation_labels" in lowered or "test_labels" in lowered:
        errors.append("treatment.py may not reference evaluation labels")
    return list(dict.fromkeys(errors))


def _benchmark_mapping(path: Path) -> dict[str, Any]:
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    mapping = value.get("benchmark", value)
    if not isinstance(mapping, Mapping):
        raise TypeError("benchmark config must be a mapping")
    return dict(mapping)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    import os
    import tempfile

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(
                dict(value),
                stream,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _merge_usage(target: dict[str, Any], current: Mapping[str, Any]) -> None:
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        target[name] = int(target.get(name, 0) or 0) + int(current.get(name, 0) or 0)
    if current.get("model"):
        target["model"] = current["model"]
