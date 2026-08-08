"""Pinned CIFAR-10 calibration benchmark with a narrow treatment API.

The adapter owns the dataset, frozen model, corruption, split, baseline,
metrics and result contract.  A generated treatment may only transform logits
after fitting on the calibration split; it never loads data or models.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import random
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol
from urllib.parse import urlparse

import numpy as np
import yaml

RESULT_SCHEMA_VERSION = 1
DEFAULT_DATASET_URL = (
    "https://hf-mirror.com/datasets/uoft-cs/cifar10/resolve/main/"
    "plain_text/test-00000-of-00001.parquet"
)
DEFAULT_WEIGHTS_URL = (
    "https://github.com/chenyaofo/pytorch-cifar-models/releases/download/"
    "resnet/cifar10_resnet20-4118986f.pt"
)
DEFAULT_MODEL_SOURCE_REPO = "https://github.com/chenyaofo/pytorch-cifar-models.git"
DEFAULT_MODEL_SOURCE_COMMIT = "786c16252c0fc58ee9adac063f8337cc4a7a497a"
DEFAULT_MODEL_NAME = "cifar10_resnet20"


class ContractError(ValueError):
    """Raised when a treatment or asset violates the benchmark contract."""


class Treatment(Protocol):
    def fit(
        self, calibration_logits: np.ndarray, calibration_labels: np.ndarray
    ) -> Any:
        """Fit only on the provided calibration split and return state."""

    def transform(self, logits: np.ndarray, state: Any) -> np.ndarray:
        """Return transformed logits with exactly the input shape."""


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    cache_dir: Path
    output_dir: Path
    treatment_path: Path
    dataset_url: str = DEFAULT_DATASET_URL
    dataset_sha256: str = ""
    weights_url: str = DEFAULT_WEIGHTS_URL
    weights_sha256: str = ""
    model_source_repo: str = DEFAULT_MODEL_SOURCE_REPO
    model_source_commit: str = DEFAULT_MODEL_SOURCE_COMMIT
    model_source_archive: Path | None = None
    model_source_archive_sha256: str = ""
    model_name: str = DEFAULT_MODEL_NAME
    dataset_format: str = "parquet"
    examples: int = 1000
    calibration_examples: int = 1000
    seeds: tuple[int, ...] = (17, 29, 43)
    corruption: str = "gaussian_noise"
    corruption_severity: float = 0.12
    batch_size: int = 256
    ece_bins: int = 15
    device: str = "cuda"
    require_cuda: bool = True
    allow_downloads: bool = True

    @classmethod
    def from_file(cls, path: str | Path) -> BenchmarkConfig:
        config_path = Path(path).expanduser().resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        value = raw.get("benchmark", raw)
        if not isinstance(value, dict):
            raise TypeError("benchmark configuration must be a mapping")

        def resolved(name: str, default: str) -> Path:
            candidate = Path(str(value.get(name, default))).expanduser()
            if not candidate.is_absolute():
                candidate = config_path.parent / candidate
            return candidate.resolve()

        seeds_raw = value.get("seeds", (17, 29, 43))
        if isinstance(seeds_raw, int):
            seeds = tuple(range(seeds_raw))
        else:
            seeds = tuple(int(item) for item in seeds_raw)
        return cls(
            cache_dir=resolved("cache_dir", "cache"),
            output_dir=resolved("output_dir", "output"),
            treatment_path=resolved("treatment_path", "treatment.py"),
            dataset_url=str(value.get("dataset_url", DEFAULT_DATASET_URL)),
            dataset_sha256=str(value.get("dataset_sha256", "") or "").lower(),
            weights_url=str(value.get("weights_url", DEFAULT_WEIGHTS_URL)),
            weights_sha256=str(value.get("weights_sha256", "") or "").lower(),
            model_source_repo=str(
                value.get("model_source_repo", DEFAULT_MODEL_SOURCE_REPO)
            ),
            model_source_commit=str(
                value.get("model_source_commit", DEFAULT_MODEL_SOURCE_COMMIT)
            ),
            model_source_archive=(
                resolved("model_source_archive", "")
                if value.get("model_source_archive")
                else None
            ),
            model_source_archive_sha256=str(
                value.get("model_source_archive_sha256", "") or ""
            ).lower(),
            model_name=str(value.get("model_name", DEFAULT_MODEL_NAME)),
            dataset_format=str(value.get("dataset_format", "parquet")),
            examples=max(1, int(value.get("examples", 1000))),
            calibration_examples=max(1, int(value.get("calibration_examples", 1000))),
            seeds=seeds or (17,),
            corruption=str(value.get("corruption", "gaussian_noise")),
            corruption_severity=float(value.get("corruption_severity", 0.12)),
            batch_size=max(1, int(value.get("batch_size", 256))),
            ece_bins=max(2, int(value.get("ece_bins", 15))),
            device=str(value.get("device", "cuda")),
            require_cuda=bool(value.get("require_cuda", True)),
            allow_downloads=bool(value.get("allow_downloads", True)),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    status: str
    metrics: dict[str, float]
    uncertainty: dict[str, float]
    per_seed: list[dict[str, Any]]
    assets: dict[str, Any]
    usage: dict[str, Any]
    provenance: dict[str, Any]
    artifacts: list[str]
    error: str = ""
    schema_version: int = RESULT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def evaluate_probabilities(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int = 15,
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.ndim != 2 or labels.shape != (probabilities.shape[0],):
        raise ContractError("probabilities/labels have incompatible shapes")
    if not np.isfinite(probabilities).all():
        raise ContractError("probabilities contain non-finite values")
    row_sums = probabilities.sum(axis=1)
    if np.any(probabilities < 0) or not np.allclose(row_sums, 1.0, atol=1e-6):
        raise ContractError("treatment output does not define probabilities")
    prediction = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (prediction == labels).astype(np.float64)
    accuracy = float(correct.mean())
    nll = float(
        -np.log(
            np.clip(probabilities[np.arange(len(labels)), labels], 1e-12, 1.0)
        ).mean()
    )
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for index in range(bins):
        lower, upper = edges[index], edges[index + 1]
        if index == bins - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)
        if mask.any():
            ece += float(mask.mean()) * abs(
                float(correct[mask].mean()) - float(confidence[mask].mean())
            )
    return {"ece": float(ece), "nll": nll, "accuracy": accuracy}


def _temperature_fit(logits: np.ndarray, labels: np.ndarray) -> float:
    # A deterministic bounded one-dimensional search is sufficient for this
    # fixed baseline and avoids adding scipy as a runtime dependency.
    candidates = np.exp(np.linspace(math.log(0.05), math.log(10.0), 401))
    best_temperature = 1.0
    best_nll = math.inf
    for temperature in candidates:
        metrics = evaluate_probabilities(
            _softmax(logits / temperature),
            labels,
            bins=15,
        )
        if metrics["nll"] < best_nll:
            best_nll = metrics["nll"]
            best_temperature = float(temperature)
    return best_temperature


def _load_treatment(path: Path) -> Treatment:
    if not path.is_file():
        raise ContractError(f"treatment module does not exist: {path}")
    spec = importlib.util.spec_from_file_location("generated_treatment", path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import treatment module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    treatment = _treatment_from_module(module)
    return treatment


def _treatment_from_module(module: ModuleType) -> Treatment:
    factory = getattr(module, "build_treatment", None)
    if not callable(factory):
        raise ContractError("treatment.py must expose build_treatment()")
    treatment = factory()
    if not callable(getattr(treatment, "fit", None)):
        raise ContractError("treatment must expose fit(logits, labels)")
    if not callable(getattr(treatment, "transform", None)):
        raise ContractError("treatment must expose transform(logits, state)")
    return treatment


def _verify_hash(path: Path, expected: str, label: str) -> str:
    observed = sha256_path(path)
    if expected and observed != expected:
        raise ContractError(
            f"{label} SHA256 mismatch: expected {expected}, observed {observed}"
        )
    return observed


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ResearchClaw-BenchmarkAdapter/1.0"},
    )
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as stream,
        ):
            shutil.copyfileobj(response, stream)
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _ensure_file(
    *,
    url: str,
    path: Path,
    expected_sha256: str,
    allow_downloads: bool,
    label: str,
) -> str:
    if not path.is_file():
        if not allow_downloads:
            raise ContractError(f"{label} missing and downloads are disabled: {path}")
        _download(url, path)
    try:
        return _verify_hash(path, expected_sha256, label)
    except ContractError:
        if not allow_downloads:
            raise
        path.unlink(missing_ok=True)
        _download(url, path)
        return _verify_hash(path, expected_sha256, label)


def _ensure_model_source(config: BenchmarkConfig) -> tuple[Path, str]:
    root = config.cache_dir / "model-source" / config.model_source_commit
    marker = root / ".researchclaw-commit"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip() == (
        config.model_source_commit
    ):
        return root, config.model_source_commit
    if root.exists():
        shutil.rmtree(root)
    if config.model_source_archive is not None:
        archive = config.model_source_archive
        _verify_hash(
            archive,
            config.model_source_archive_sha256,
            "model source archive",
        )
        temporary = root.with_name(root.name + ".partial")
        shutil.rmtree(temporary, ignore_errors=True)
        temporary.mkdir(parents=True, exist_ok=True)
        with tarfile.open(archive, "r:gz") as stream:
            members = stream.getmembers()
            if any(
                member.name.startswith("/") or ".." in Path(member.name).parts
                for member in members
            ):
                raise ContractError("unsafe path in model source archive")
            stream.extractall(temporary)
        children = [item for item in temporary.iterdir() if item.is_dir()]
        source = (
            children[0]
            if len(children) == 1 and (children[0] / "pytorch_cifar_models").is_dir()
            else temporary
        )
        observed = (
            subprocess.check_output(
                ["git", "-C", str(source), "rev-parse", "HEAD"],
                text=True,
                timeout=30,
            )
            .strip()
            .lower()
        )
        if observed != config.model_source_commit.lower():
            raise ContractError(
                f"model source commit mismatch: expected "
                f"{config.model_source_commit}, observed {observed}"
            )
        if source != temporary:
            source.replace(root)
            shutil.rmtree(temporary, ignore_errors=True)
        else:
            temporary.replace(root)
        (root / ".researchclaw-commit").write_text(
            observed + "\n",
            encoding="utf-8",
        )
        return root, observed
    if not config.allow_downloads:
        raise ContractError(f"pinned model source is missing: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    temporary = root.with_name(root.name + ".partial")
    shutil.rmtree(temporary, ignore_errors=True)
    subprocess.run(
        ["git", "clone", "--quiet", config.model_source_repo, str(temporary)],
        check=True,
        timeout=180,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(temporary),
            "checkout",
            "--quiet",
            config.model_source_commit,
        ],
        check=True,
        timeout=60,
    )
    observed = (
        subprocess.check_output(
            ["git", "-C", str(temporary), "rev-parse", "HEAD"],
            text=True,
            timeout=30,
        )
        .strip()
        .lower()
    )
    if observed != config.model_source_commit.lower():
        raise ContractError(
            f"model source commit mismatch: expected "
            f"{config.model_source_commit}, observed {observed}"
        )
    (temporary / ".researchclaw-commit").write_text(observed + "\n", encoding="utf-8")
    temporary.replace(root)
    return root, observed


def _ensure_dataset(config: BenchmarkConfig) -> tuple[Path, dict[str, str]]:
    suffix = ".parquet" if config.dataset_format == "parquet" else ".tar.gz"
    dataset_path = config.cache_dir / "datasets" / f"cifar10-test{suffix}"
    dataset_hash = _ensure_file(
        url=config.dataset_url,
        path=dataset_path,
        expected_sha256=config.dataset_sha256,
        allow_downloads=config.allow_downloads,
        label="CIFAR-10 test dataset",
    )
    if config.dataset_format == "parquet":
        return dataset_path, {
            "dataset_format": "parquet",
            "dataset_sha256": dataset_hash,
        }
    if config.dataset_format != "python_tar":
        raise ContractError(f"unsupported dataset_format: {config.dataset_format}")
    extracted = config.cache_dir / "datasets" / "cifar-10-batches-py"
    if not (extracted / "test_batch").is_file():
        if extracted.exists():
            shutil.rmtree(extracted)
        with tarfile.open(dataset_path, "r:gz") as stream:
            members = stream.getmembers()
            if any(
                member.name.startswith("/") or ".." in Path(member.name).parts
                for member in members
            ):
                raise ContractError("unsafe path in CIFAR-10 archive")
            stream.extractall(config.cache_dir / "datasets")
    return extracted, {
        "dataset_format": "python_tar",
        "dataset_sha256": dataset_hash,
    }


def _load_cifar10_test(
    path: Path,
    *,
    dataset_format: str,
) -> tuple[np.ndarray, np.ndarray]:
    if dataset_format == "parquet":
        try:
            from pyarrow import parquet
        except ImportError as exc:
            raise ContractError(
                "pyarrow is required to load the pinned CIFAR-10 parquet"
            ) from exc
        from io import BytesIO

        from PIL import Image

        table = parquet.read_table(path, columns=["img", "label"])
        labels = np.asarray(table.column("label").to_pylist(), dtype=np.int64)
        images = np.empty((len(labels), 3, 32, 32), dtype=np.uint8)
        for index, value in enumerate(table.column("img").to_pylist()):
            if isinstance(value, dict):
                encoded = value.get("bytes")
                if encoded is None and value.get("path"):
                    image = Image.open(value["path"])
                else:
                    image = Image.open(BytesIO(encoded))
            elif isinstance(value, (bytes, bytearray)):
                image = Image.open(BytesIO(value))
            else:
                raise ContractError(
                    f"unsupported parquet image value: {type(value).__name__}"
                )
            array = np.asarray(image.convert("RGB"), dtype=np.uint8)
            images[index] = np.moveaxis(array, -1, 0)
        return images, labels
    if dataset_format != "python_tar":
        raise ContractError(f"unsupported dataset_format: {dataset_format}")
    import pickle

    with (path / "test_batch").open("rb") as stream:
        batch = pickle.load(stream, encoding="bytes")
    flat = np.asarray(batch[b"data"], dtype=np.uint8)
    labels = np.asarray(batch[b"labels"], dtype=np.int64)
    images = flat.reshape(-1, 3, 32, 32)
    return images, labels


def _corrupt(
    images: np.ndarray,
    *,
    seed: int,
    corruption: str,
    severity: float,
) -> np.ndarray:
    values = images.astype(np.float32) / 255.0
    rng = np.random.default_rng(seed)
    if corruption == "gaussian_noise":
        values = values + rng.normal(0.0, severity, size=values.shape).astype(
            np.float32
        )
    elif corruption == "brightness":
        values = values + severity
    elif corruption == "contrast":
        mean = values.mean(axis=(2, 3), keepdims=True)
        values = mean + (1.0 + severity) * (values - mean)
    else:
        raise ContractError(f"unsupported corruption: {corruption}")
    return np.clip(values, 0.0, 1.0)


class Cifar10CalibrationAdapter:
    def __init__(self, config: BenchmarkConfig) -> None:
        self.config = config

    def run(self) -> BenchmarkResult:
        started = time.monotonic()
        import torch

        if self.config.require_cuda and not torch.cuda.is_available():
            raise ContractError("CUDA is required but unavailable")
        device = torch.device(
            self.config.device
            if self.config.device != "cuda" or torch.cuda.is_available()
            else "cpu"
        )
        dataset_root, dataset_assets = _ensure_dataset(self.config)
        weights = (
            self.config.cache_dir
            / "models"
            / Path(urlparse(self.config.weights_url).path).name
        )
        weights_hash = _ensure_file(
            url=self.config.weights_url,
            path=weights,
            expected_sha256=self.config.weights_sha256,
            allow_downloads=self.config.allow_downloads,
            label="pretrained model weights",
        )
        source_root, source_commit = _ensure_model_source(self.config)
        sys.path.insert(0, str(source_root))
        try:
            package = __import__("pytorch_cifar_models")
            factory = getattr(package, self.config.model_name)
            model = factory(pretrained=False)
        finally:
            try:
                sys.path.remove(str(source_root))
            except ValueError:
                pass
        state_dict = torch.load(weights, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval().to(device)

        images, labels = _load_cifar10_test(
            dataset_root,
            dataset_format=self.config.dataset_format,
        )
        required = self.config.calibration_examples + self.config.examples
        if required > len(labels):
            raise ContractError(
                f"requested {required} CIFAR-10 examples but only {len(labels)} exist"
            )
        treatment = _load_treatment(self.config.treatment_path)
        rows: list[dict[str, Any]] = []
        for seed in self.config.seeds:
            order = np.arange(len(labels))
            random.Random(seed).shuffle(order)
            calibration_indices = order[: self.config.calibration_examples]
            evaluation_indices = order[self.config.calibration_examples : required]
            calibration_logits = self._infer(
                model,
                _corrupt(
                    images[calibration_indices],
                    seed=seed * 2 + 1,
                    corruption=self.config.corruption,
                    severity=self.config.corruption_severity,
                ),
                device=device,
                torch=torch,
            )
            evaluation_logits = self._infer(
                model,
                _corrupt(
                    images[evaluation_indices],
                    seed=seed * 2 + 2,
                    corruption=self.config.corruption,
                    severity=self.config.corruption_severity,
                ),
                device=device,
                torch=torch,
            )
            calibration_labels = labels[calibration_indices]
            evaluation_labels = labels[evaluation_indices]

            temperature = _temperature_fit(calibration_logits, calibration_labels)
            baseline_metrics = evaluate_probabilities(
                _softmax(evaluation_logits / temperature),
                evaluation_labels,
                bins=self.config.ece_bins,
            )
            state = treatment.fit(
                calibration_logits.copy(),
                calibration_labels.copy(),
            )
            transformed = np.asarray(
                treatment.transform(evaluation_logits.copy(), state),
                dtype=np.float64,
            )
            if transformed.shape != evaluation_logits.shape:
                raise ContractError(
                    "treatment transformed logits must preserve shape "
                    f"{evaluation_logits.shape}, got {transformed.shape}"
                )
            if not np.isfinite(transformed).all():
                raise ContractError(
                    "treatment transformed logits contain non-finite values"
                )
            treatment_metrics = evaluate_probabilities(
                _softmax(transformed),
                evaluation_labels,
                bins=self.config.ece_bins,
            )
            rows.append(
                {
                    "seed": seed,
                    "temperature": temperature,
                    "baseline": baseline_metrics,
                    "treatment": treatment_metrics,
                    "effect_ece": (baseline_metrics["ece"] - treatment_metrics["ece"]),
                }
            )
        metrics: dict[str, float] = {}
        uncertainty: dict[str, float] = {}
        for method in ("baseline", "treatment"):
            for metric in ("ece", "nll", "accuracy"):
                values = np.asarray(
                    [row[method][metric] for row in rows],
                    dtype=np.float64,
                )
                metrics[f"{method}_{metric}"] = float(values.mean())
                uncertainty[f"{method}_{metric}_std"] = float(
                    values.std(ddof=1) if len(values) > 1 else 0.0
                )
        effects = np.asarray([row["effect_ece"] for row in rows], dtype=np.float64)
        metrics["effect_ece"] = float(effects.mean())
        uncertainty["effect_ece_std"] = float(
            effects.std(ddof=1) if len(effects) > 1 else 0.0
        )
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        result_path = self.config.output_dir / "result.json"
        result = BenchmarkResult(
            status="ok",
            metrics=metrics,
            uncertainty=uncertainty,
            per_seed=rows,
            assets={
                **dataset_assets,
                "weights_sha256": weights_hash,
                "model_source_commit": source_commit,
                "model_name": self.config.model_name,
            },
            usage={
                "device": str(device),
                "gpu_count": int(device.type == "cuda"),
                "wall_seconds": time.monotonic() - started,
                "examples_per_seed": self.config.examples,
                "calibration_examples_per_seed": self.config.calibration_examples,
                "seeds": len(self.config.seeds),
            },
            provenance={
                "adapter": "Cifar10CalibrationAdapter",
                "adapter_schema_version": RESULT_SCHEMA_VERSION,
                "treatment_path": str(self.config.treatment_path),
                "treatment_sha256": sha256_path(self.config.treatment_path),
                "corruption": self.config.corruption,
                "corruption_severity": self.config.corruption_severity,
            },
            artifacts=[str(result_path)],
        )
        result_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return result

    def _infer(
        self, model: Any, images: np.ndarray, *, device: Any, torch: Any
    ) -> np.ndarray:
        mean = torch.tensor(
            [0.4914, 0.4822, 0.4465],
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            [0.2023, 0.1994, 0.2010],
            dtype=torch.float32,
            device=device,
        ).view(1, 3, 1, 1)
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for offset in range(0, len(images), self.config.batch_size):
                batch = torch.from_numpy(
                    images[offset : offset + self.config.batch_size]
                ).to(device=device, dtype=torch.float32)
                logits = model((batch - mean) / std)
                outputs.append(logits.detach().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float64)


def run_from_file(path: str | Path) -> BenchmarkResult:
    config = BenchmarkConfig.from_file(path)
    return Cifar10CalibrationAdapter(config).run()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="researchclaw-benchmark-cifar10",
        description="Run the pinned CIFAR-10 calibration benchmark adapter",
    )
    parser.add_argument("-c", "--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = run_from_file(args.config)
    except Exception as exc:  # noqa: BLE001
        config = BenchmarkConfig.from_file(args.config)
        config.output_dir.mkdir(parents=True, exist_ok=True)
        failed = BenchmarkResult(
            status="error",
            metrics={},
            uncertainty={},
            per_seed=[],
            assets={},
            usage={},
            provenance={"adapter": "Cifar10CalibrationAdapter"},
            artifacts=[],
            error=f"{type(exc).__name__}: {exc}",
        )
        (config.output_dir / "result.json").write_text(
            json.dumps(failed.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failed.to_dict(), ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
