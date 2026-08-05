"""Trusted execution contracts and controller-signed attestations.

Generated experiment code is untrusted.  It may write scientific artifacts,
but it must not be trusted to describe what command ran, what resources were
allocated, or which bytes were produced.  This module keeps those claims in a
small controller-owned contract/attestation format and binds them with
content hashes plus an HMAC signature.

The signing key must remain in the controller (or another trusted wrapper)
and must never be passed through the experiment environment.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import stat
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Any

MANIFEST_SCHEMA = "autoresearch_v2.sha256_manifest"
CONTRACT_SCHEMA = "autoresearch_v2.execution_contract"
ATTESTATION_SCHEMA = "autoresearch_v2.execution_attestation"
SCHEMA_VERSION = 1
DEFAULT_ATTESTATION_EXCLUDES = ("execution_attestation.json",)

_HEX_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_RESOURCE_KEY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

_MANIFEST_KEYS = {
    "schema",
    "version",
    "algorithm",
    "root",
    "excluded_paths",
    "files",
}
_FILE_RECORD_KEYS = {"path", "sha256", "size_bytes"}
_CONTRACT_KEYS = {
    "schema",
    "version",
    "idea_id",
    "job_id",
    "attempt_id",
    "mode",
    "argv",
    "cwd",
    "entrypoint",
    "output_dir",
    "resource_limits",
    "input_hashes",
    "allowed_env_keys",
    "artifact_excludes",
}
_INPUT_HASH_KEYS = {"plan", "build"}
_ATTESTATION_KEYS = {
    "schema",
    "version",
    "idea_id",
    "job_id",
    "attempt_id",
    "mode",
    "contract_hash",
    "execution",
    "logs",
    "artifact_manifest",
    "artifact_manifest_hash",
    "authentication",
}
_EXECUTION_KEYS = {
    "started_at",
    "ended_at",
    "returncode",
    "allocated_gpus",
}
_LOG_KEYS = {"stdout", "stderr"}
_LOG_RECORD_KEYS = {"sha256", "size_bytes"}
_AUTH_KEYS = {"algorithm", "key_id", "signature"}


class AttestationError(ValueError):
    """Base error for unsafe paths or malformed trusted-execution data."""


class AttestationValidationError(AttestationError):
    """Raised when a contract, manifest, or attestation fails verification."""

    def __init__(self, errors: Iterable[str]) -> None:
        self.errors = tuple(str(error) for error in errors)
        super().__init__("; ".join(self.errors))


def canonical_json_bytes(value: Any) -> bytes:
    """Return deterministic UTF-8 JSON bytes suitable for hashing/signing."""

    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise AttestationError(f"value is not canonical JSON: {exc}") from exc
    return text.encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Hash a JSON value using the canonical encoding above."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash one regular file without following a final-component symlink."""

    return _file_record(Path(path))["sha256"]


def build_tree_manifest(
    root: str | Path,
    *,
    exclude: Iterable[str | Path] = DEFAULT_ATTESTATION_EXCLUDES,
) -> dict[str, Any]:
    """Recursively hash regular files below ``root``.

    Exclusions are exact root-relative paths or directory prefixes.  Globs are
    deliberately unsupported so the controller and verifier cannot interpret
    an exclusion differently.  Symlinks and non-regular files fail closed.
    """

    root_path = _require_directory(Path(root))
    excluded = _normalize_excludes(exclude)
    files: list[dict[str, Any]] = []

    for current, directories, filenames in os.walk(
        root_path,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in sorted(directories):
            path = current_path / name
            relative = _relative_posix(path, root_path)
            if _is_excluded(relative, excluded):
                continue
            if path.is_symlink():
                raise AttestationError(
                    f"symlink directories are forbidden: {relative}"
                )
            if not path.is_dir():
                raise AttestationError(
                    f"non-directory encountered during traversal: {relative}"
                )
            kept_directories.append(name)
        directories[:] = kept_directories

        for name in sorted(filenames):
            path = current_path / name
            relative = _relative_posix(path, root_path)
            if _is_excluded(relative, excluded):
                continue
            record = _file_record(path)
            files.append({"path": relative, **record})

    files.sort(key=lambda item: str(item["path"]))
    return {
        "schema": MANIFEST_SCHEMA,
        "version": SCHEMA_VERSION,
        "algorithm": "sha256",
        "root": ".",
        "excluded_paths": list(excluded),
        "files": files,
    }


def build_sha256_manifest(
    root: str | Path,
    *,
    exclude_paths: Iterable[str | Path] = DEFAULT_ATTESTATION_EXCLUDES,
) -> dict[str, Any]:
    """Backward-compatible spelling for :func:`build_tree_manifest`."""

    return build_tree_manifest(root, exclude=exclude_paths)


def verify_sha256_manifest(
    root: str | Path,
    manifest: Mapping[str, Any],
    *,
    exclude_paths: Iterable[str | Path] | None = None,
) -> str:
    """Strictly verify a manifest and return its canonical SHA256."""

    errors = _manifest_schema_errors(manifest)
    if errors:
        raise AttestationValidationError(errors)
    declared_excludes = tuple(str(item) for item in manifest["excluded_paths"])
    expected_excludes = (
        _normalize_excludes(exclude_paths)
        if exclude_paths is not None
        else declared_excludes
    )
    if declared_excludes != expected_excludes:
        raise AttestationValidationError(
            [
                (
                    "manifest excluded_paths do not match the trusted "
                    "exclusion policy"
                )
            ]
        )
    actual = build_tree_manifest(root, exclude=expected_excludes)
    if dict(manifest) != actual:
        raise AttestationValidationError(
            ["artifact manifest does not match current filesystem contents"]
        )
    return canonical_json_sha256(actual)


def create_execution_contract(
    *,
    idea_id: str,
    job_id: str,
    attempt_id: str,
    mode: str,
    argv: Sequence[str],
    cwd: str | Path,
    entrypoint: str | Path,
    output_dir: str | Path,
    resource_limits: Mapping[str, Any],
    plan_path: str | Path,
    build_path: str | Path,
    allowed_env_keys: Iterable[str],
    artifact_exclude_paths: Iterable[
        str | Path
    ] = DEFAULT_ATTESTATION_EXCLUDES,
) -> dict[str, Any]:
    """Construct an immutable execution contract from trusted controller data."""

    cwd_path = _require_directory(Path(cwd))
    normalized_argv = _normalize_argv(argv)
    normalized_entrypoint = _path_relative_to_root(
        entrypoint,
        root=cwd_path,
        require_file=True,
        label="entrypoint",
    )
    normalized_output = _path_relative_to_root(
        output_dir,
        root=cwd_path,
        require_file=False,
        label="output_dir",
    )
    plan_record = _bound_input_record(
        plan_path,
        root=cwd_path,
        label="plan_path",
    )
    build_record = _bound_input_record(
        build_path,
        root=cwd_path,
        label="build_path",
    )
    contract = {
        "schema": CONTRACT_SCHEMA,
        "version": SCHEMA_VERSION,
        "idea_id": _required_text(idea_id, "idea_id"),
        "job_id": _required_text(job_id, "job_id"),
        "attempt_id": _required_text(attempt_id, "attempt_id"),
        "mode": _required_text(mode, "mode"),
        "argv": normalized_argv,
        "cwd": ".",
        "entrypoint": normalized_entrypoint,
        "output_dir": normalized_output,
        "resource_limits": _normalize_resource_limits(resource_limits),
        "input_hashes": {
            "plan": plan_record,
            "build": build_record,
        },
        "allowed_env_keys": _normalize_env_keys(allowed_env_keys),
        "artifact_excludes": list(
            _normalize_excludes(artifact_exclude_paths)
        ),
    }
    errors = _contract_schema_errors(contract)
    if errors:
        raise AttestationValidationError(errors)
    return contract


def build_execution_contract(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible spelling for :func:`create_execution_contract`."""

    return create_execution_contract(**kwargs)


def write_execution_contract(
    path: str | Path,
    contract: Mapping[str, Any] | None = None,
    **contract_inputs: Any,
) -> dict[str, Any]:
    """Create or accept, atomically write, and return an execution contract.

    ``argv`` remains a structured ``list[str]`` throughout; callers must not
    pass an opaque shell string.
    """

    if contract is not None and contract_inputs:
        raise AttestationError(
            "pass either contract or contract construction inputs, not both"
        )
    value = (
        dict(contract)
        if contract is not None
        else create_execution_contract(**contract_inputs)
    )
    errors = _contract_schema_errors(value)
    if errors:
        raise AttestationValidationError(errors)
    _atomic_write_json(Path(path), value)
    return value


def verify_execution_contract(
    contract: Mapping[str, Any],
    *,
    idea_id: str,
    job_id: str,
    attempt_id: str,
    mode: str,
    argv: Sequence[str],
    cwd: str | Path,
    entrypoint: str | Path,
    output_dir: str | Path,
    resource_limits: Mapping[str, Any],
    plan_path: str | Path,
    build_path: str | Path,
    allowed_env_keys: Iterable[str],
    artifact_exclude_paths: Iterable[
        str | Path
    ] = DEFAULT_ATTESTATION_EXCLUDES,
) -> str:
    """Rebuild and strictly compare a contract against trusted inputs."""

    errors = _contract_schema_errors(contract)
    if errors:
        raise AttestationValidationError(errors)
    expected = create_execution_contract(
        idea_id=idea_id,
        job_id=job_id,
        attempt_id=attempt_id,
        mode=mode,
        argv=argv,
        cwd=cwd,
        entrypoint=entrypoint,
        output_dir=output_dir,
        resource_limits=resource_limits,
        plan_path=plan_path,
        build_path=build_path,
        allowed_env_keys=allowed_env_keys,
        artifact_exclude_paths=artifact_exclude_paths,
    )
    if dict(contract) != expected:
        raise AttestationValidationError(
            ["execution contract does not match trusted controller inputs"]
        )
    return canonical_json_sha256(expected)


def load_execution_contract(
    path: str | Path,
    **trusted_inputs: Any,
) -> dict[str, Any]:
    """Load and strictly verify a contract written by the controller."""

    value = _read_json_object(Path(path))
    verify_execution_contract(value, **trusted_inputs)
    return value


def create_execution_attestation(
    contract: Mapping[str, Any],
    *,
    signing_key: bytes,
    key_id: str,
    started_at: str | datetime,
    ended_at: str | datetime,
    returncode: int,
    allocated_gpus: int,
    stdout_path: str | Path,
    stderr_path: str | Path,
    artifact_dir: str | Path,
) -> dict[str, Any]:
    """Create a controller-signed attestation over logs and output artifacts."""

    contract_errors = _contract_schema_errors(contract)
    if contract_errors:
        raise AttestationValidationError(contract_errors)
    key = _require_signing_key(signing_key)
    start = _normalize_timestamp(started_at, "started_at")
    end = _normalize_timestamp(ended_at, "ended_at")
    if _parse_timestamp(end) < _parse_timestamp(start):
        raise AttestationError("ended_at must not precede started_at")
    gpu_count = _nonnegative_int(allocated_gpus, "allocated_gpus")
    code = _strict_int(returncode, "returncode")
    excludes = tuple(str(item) for item in contract["artifact_excludes"])
    artifact_manifest = build_tree_manifest(
        artifact_dir,
        exclude=excludes,
    )

    unsigned: dict[str, Any] = {
        "schema": ATTESTATION_SCHEMA,
        "version": SCHEMA_VERSION,
        "idea_id": contract["idea_id"],
        "job_id": contract["job_id"],
        "attempt_id": contract["attempt_id"],
        "mode": contract["mode"],
        "contract_hash": canonical_json_sha256(contract),
        "execution": {
            "started_at": start,
            "ended_at": end,
            "returncode": code,
            "allocated_gpus": gpu_count,
        },
        "logs": {
            "stdout": _file_record(Path(stdout_path)),
            "stderr": _file_record(Path(stderr_path)),
        },
        "artifact_manifest": artifact_manifest,
        "artifact_manifest_hash": canonical_json_sha256(artifact_manifest),
        "authentication": {
            "algorithm": "hmac-sha256",
            "key_id": _required_text(key_id, "key_id"),
        },
    }
    signature = hmac.new(
        key,
        canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()
    attestation = _deep_copy_json(unsigned)
    attestation["authentication"]["signature"] = signature
    errors = _attestation_schema_errors(attestation)
    if errors:
        raise AttestationValidationError(errors)
    return attestation


def build_execution_attestation(
    contract: Mapping[str, Any],
    **kwargs: Any,
) -> dict[str, Any]:
    """Backward-compatible spelling for :func:`create_execution_attestation`."""

    return create_execution_attestation(contract, **kwargs)


def write_execution_attestation(
    path: str | Path,
    attestation: Mapping[str, Any],
) -> str:
    """Atomically write a validated attestation and return its JSON hash."""

    errors = _attestation_schema_errors(attestation)
    if errors:
        raise AttestationValidationError(errors)
    _atomic_write_json(Path(path), attestation)
    return canonical_json_sha256(attestation)


def verify_execution_attestation_or_raise(
    attestation: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    signing_key: bytes,
    key_id: str,
    started_at: str | datetime,
    ended_at: str | datetime,
    returncode: int,
    allocated_gpus: int,
    stdout_path: str | Path,
    stderr_path: str | Path,
    artifact_dir: str | Path,
    attestation_path: str | Path | None = None,
) -> str:
    """Strictly verify controller provenance and all bound execution bytes.

    Runtime values are mandatory trusted inputs; values asserted by generated
    code are never accepted as their own evidence.
    """

    errors = _contract_schema_errors(contract)
    errors.extend(_attestation_schema_errors(attestation))
    if errors:
        raise AttestationValidationError(errors)

    key = _require_signing_key(signing_key)
    expected_key_id = _required_text(key_id, "key_id")
    authentication = dict(attestation["authentication"])
    signature = str(authentication.pop("signature"))
    unsigned = _deep_copy_json(attestation)
    unsigned["authentication"] = authentication
    expected_signature = hmac.new(
        key,
        canonical_json_bytes(unsigned),
        hashlib.sha256,
    ).hexdigest()
    if not hmac.compare_digest(signature, expected_signature):
        raise AttestationValidationError(
            ["execution attestation signature is invalid"]
        )
    if authentication["key_id"] != expected_key_id:
        raise AttestationValidationError(
            ["execution attestation key_id does not match trusted key"]
        )

    expected = create_execution_attestation(
        contract,
        signing_key=key,
        key_id=expected_key_id,
        started_at=started_at,
        ended_at=ended_at,
        returncode=returncode,
        allocated_gpus=allocated_gpus,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        artifact_dir=artifact_dir,
    )
    if dict(attestation) != expected:
        raise AttestationValidationError(
            [
                (
                    "execution attestation does not match trusted execution "
                    "inputs or current filesystem contents"
                )
            ]
        )

    if attestation_path is not None:
        trusted_path = Path(attestation_path)
        artifact_root = _require_directory(Path(artifact_dir))
        resolved_attestation = trusted_path.resolve(strict=True)
        if resolved_attestation == artifact_root or resolved_attestation.is_relative_to(
            artifact_root
        ):
            raise AttestationValidationError(
                ["execution attestation must be stored outside artifact_dir"]
            )
        on_disk = _read_json_object(trusted_path)
        if on_disk != dict(attestation):
            raise AttestationValidationError(
                ["on-disk execution attestation differs from verified value"]
            )

    return canonical_json_sha256(expected)


def verify_execution_attestation(
    contract_path: str | Path,
    attestation_path: str | Path,
    root: str | Path,
    *,
    signing_key: bytes,
    key_id: str,
    stdout_path: str | Path,
    stderr_path: str | Path,
    artifact_dir: str | Path | None = None,
    started_at: str | datetime | None = None,
    ended_at: str | datetime | None = None,
    returncode: int | None = None,
    allocated_gpus: int | None = None,
) -> list[str]:
    """Verify a persisted contract/attestation pair, returning all errors.

    This is the controller/jobs integration API.  ``root`` is the candidate
    project root used as contract ``cwd``.  When trusted runtime values are
    supplied they must match exactly; otherwise their signed attestation
    values are used.  The signing key, current logs, current artifact tree,
    current plan/build files, and contract-bound paths are always verified.
    """

    errors: list[str] = []
    try:
        contract = _read_json_object(Path(contract_path))
        attestation = _read_json_object(Path(attestation_path))
        contract_errors = _contract_schema_errors(contract)
        if contract_errors:
            raise AttestationValidationError(contract_errors)
        attestation_errors = _attestation_schema_errors(attestation)
        if attestation_errors:
            raise AttestationValidationError(attestation_errors)

        project_root = _require_directory(Path(root))
        _verify_contract_bound_files(project_root, contract)
        execution = attestation["execution"]
        resolved_artifacts = (
            Path(artifact_dir)
            if artifact_dir is not None
            else project_root / str(contract["output_dir"])
        )
        verify_execution_attestation_or_raise(
            attestation,
            contract=contract,
            signing_key=signing_key,
            key_id=key_id,
            started_at=(
                execution["started_at"]
                if started_at is None
                else started_at
            ),
            ended_at=(
                execution["ended_at"] if ended_at is None else ended_at
            ),
            returncode=(
                int(execution["returncode"])
                if returncode is None
                else returncode
            ),
            allocated_gpus=(
                int(execution["allocated_gpus"])
                if allocated_gpus is None
                else allocated_gpus
            ),
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifact_dir=resolved_artifacts,
            attestation_path=attestation_path,
        )
    except (AttestationError, OSError, TypeError, ValueError) as exc:
        if isinstance(exc, AttestationValidationError):
            errors.extend(exc.errors)
        else:
            errors.append(str(exc))
    return errors


def load_execution_attestation(
    path: str | Path,
    **trusted_inputs: Any,
) -> dict[str, Any]:
    """Load and strictly verify a controller-side execution attestation."""

    value = _read_json_object(Path(path))
    verify_execution_attestation_or_raise(
        value,
        attestation_path=path,
        **trusted_inputs,
    )
    return value


def _verify_contract_bound_files(
    root: Path,
    contract: Mapping[str, Any],
) -> None:
    for name in sorted(_INPUT_HASH_KEYS):
        record = contract["input_hashes"][name]
        relative = _normalize_relative_path(str(record["path"]))
        path = root / relative
        actual = _file_record(path)
        expected = {
            "sha256": record["sha256"],
            "size_bytes": record["size_bytes"],
        }
        if actual != expected:
            raise AttestationValidationError(
                [f"contract-bound {name} file has changed"]
            )
    entrypoint = root / _normalize_relative_path(
        str(contract["entrypoint"])
    )
    _file_record(entrypoint)
    output_dir = (
        root / _normalize_relative_path(str(contract["output_dir"]))
    ).resolve(strict=True)
    if not output_dir.is_relative_to(root) or not output_dir.is_dir():
        raise AttestationValidationError(
            ["contract output_dir is not a safe directory below root"]
        )


def _file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise AttestationError(f"symlink files are forbidden: {path}")
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise AttestationError(f"cannot open regular file {path}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise AttestationError(f"not a regular file: {path}")
        digest = hashlib.sha256()
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        after = os.fstat(descriptor)
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise AttestationError(f"file changed while hashing: {path}")
        return {
            "sha256": digest.hexdigest(),
            "size_bytes": before.st_size,
        }
    finally:
        os.close(descriptor)


def _bound_input_record(
    path: str | Path,
    *,
    root: Path,
    label: str,
) -> dict[str, Any]:
    relative = _path_relative_to_root(
        path,
        root=root,
        require_file=True,
        label=label,
    )
    record = _file_record(root / relative)
    return {"path": relative, **record}


def _path_relative_to_root(
    value: str | Path,
    *,
    root: Path,
    require_file: bool,
    label: str,
) -> str:
    raw = Path(value)
    candidate = raw if raw.is_absolute() else root / raw
    _assert_no_symlink_components(candidate, root)
    resolved = candidate.resolve(strict=require_file)
    if not resolved.is_relative_to(root):
        raise AttestationError(f"{label} escapes cwd: {value}")
    relative = resolved.relative_to(root).as_posix()
    if relative in {"", "."}:
        raise AttestationError(f"{label} must not be cwd itself")
    _normalize_relative_path(relative)
    if require_file:
        if not resolved.is_file():
            raise AttestationError(f"{label} must be a regular file: {value}")
    else:
        _assert_existing_parents_are_safe(candidate, root)
    return relative


def _assert_existing_parents_are_safe(path: Path, root: Path) -> None:
    current = path
    while current != root:
        if (current.exists() or current.is_symlink()) and current.is_symlink():
            raise AttestationError(
                f"symlink path component is forbidden: {current}"
            )
        current = current.parent
        if not current.is_relative_to(root) and current != root:
            raise AttestationError(f"path escapes cwd: {path}")


def _assert_no_symlink_components(path: Path, root: Path) -> None:
    current = path
    while current != root:
        if current.is_symlink():
            raise AttestationError(
                f"symlink path component is forbidden: {current}"
            )
        current = current.parent


def _require_directory(path: Path) -> Path:
    if path.is_symlink():
        raise AttestationError(f"root directory must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise AttestationError(f"directory does not exist: {path}") from exc
    if not resolved.is_dir():
        raise AttestationError(f"not a directory: {path}")
    return resolved


def _relative_posix(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as exc:
        raise AttestationError(f"path escapes manifest root: {path}") from exc
    return _normalize_relative_path(relative)


def _normalize_relative_path(value: str | Path) -> str:
    text = str(value)
    if "\\" in text or "\x00" in text:
        raise AttestationError(f"unsafe relative path: {text!r}")
    path = PurePosixPath(text)
    if path.is_absolute() or text in {"", "."}:
        raise AttestationError(f"path must be non-empty and relative: {text!r}")
    if any(part in {"", ".", ".."} for part in path.parts):
        raise AttestationError(f"unsafe relative path: {text!r}")
    return path.as_posix()


def _normalize_excludes(values: Iterable[str | Path]) -> tuple[str, ...]:
    return tuple(
        sorted({_normalize_relative_path(value) for value in values})
    )


def _is_excluded(path: str, excluded: tuple[str, ...]) -> bool:
    return any(path == item or path.startswith(f"{item}/") for item in excluded)


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    if isinstance(argv, (str, bytes)) or not argv:
        raise AttestationError("argv must be a non-empty sequence of strings")
    result: list[str] = []
    for index, item in enumerate(argv):
        if not isinstance(item, str) or not item or "\x00" in item:
            raise AttestationError(f"invalid argv[{index}]")
        result.append(item)
    return result


def _normalize_env_keys(values: Iterable[str]) -> list[str]:
    result: set[str] = set()
    for value in values:
        if not isinstance(value, str) or not _ENV_KEY_RE.fullmatch(value):
            raise AttestationError(f"invalid environment key: {value!r}")
        result.add(value)
    return sorted(result)


def _normalize_resource_limits(
    values: Mapping[str, Any],
) -> dict[str, int | float]:
    if not isinstance(values, Mapping) or not values:
        raise AttestationError("resource_limits must be a non-empty mapping")
    result: dict[str, int | float] = {}
    for key, value in values.items():
        if not isinstance(key, str) or not _RESOURCE_KEY_RE.fullmatch(key):
            raise AttestationError(f"invalid resource limit key: {key!r}")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise AttestationError(f"resource limit {key} must be numeric")
        if value < 0:
            raise AttestationError(f"resource limit {key} must be nonnegative")
        if isinstance(value, float) and not value.is_integer():
            result[key] = value
        else:
            result[key] = int(value)
    return dict(sorted(result.items()))


def _manifest_schema_errors(value: Mapping[str, Any]) -> list[str]:
    errors = _exact_keys(value, _MANIFEST_KEYS, "manifest")
    if value.get("schema") != MANIFEST_SCHEMA:
        errors.append("invalid manifest schema")
    if value.get("version") != SCHEMA_VERSION:
        errors.append("invalid manifest version")
    if value.get("algorithm") != "sha256":
        errors.append("invalid manifest algorithm")
    if value.get("root") != ".":
        errors.append("manifest root must be '.'")
    excluded = value.get("excluded_paths")
    if not isinstance(excluded, list):
        errors.append("manifest excluded_paths must be a list")
    else:
        try:
            if list(_normalize_excludes(excluded)) != excluded:
                errors.append(
                    "manifest excluded_paths must be normalized and sorted"
                )
        except AttestationError as exc:
            errors.append(str(exc))
    files = value.get("files")
    if not isinstance(files, list):
        errors.append("manifest files must be a list")
        return errors
    seen: set[str] = set()
    ordered: list[str] = []
    for index, record in enumerate(files):
        if not isinstance(record, Mapping):
            errors.append(f"manifest files[{index}] must be an object")
            continue
        errors.extend(
            _exact_keys(record, _FILE_RECORD_KEYS, f"files[{index}]")
        )
        path = record.get("path")
        try:
            normalized = _normalize_relative_path(str(path))
        except AttestationError as exc:
            errors.append(str(exc))
            continue
        if normalized != path:
            errors.append(f"manifest path is not normalized: {path!r}")
        if normalized in seen:
            errors.append(f"duplicate manifest path: {normalized}")
        seen.add(normalized)
        ordered.append(normalized)
        _validate_digest_record(record, errors, f"files[{index}]")
    if ordered != sorted(ordered):
        errors.append("manifest files must be sorted by path")
    return errors


def _contract_schema_errors(value: Mapping[str, Any]) -> list[str]:
    errors = _exact_keys(value, _CONTRACT_KEYS, "contract")
    if value.get("schema") != CONTRACT_SCHEMA:
        errors.append("invalid contract schema")
    if value.get("version") != SCHEMA_VERSION:
        errors.append("invalid contract version")
    for field in ("idea_id", "job_id", "attempt_id", "mode"):
        if not isinstance(value.get(field), str) or not value.get(field):
            errors.append(f"invalid contract {field}")
    if value.get("cwd") != ".":
        errors.append("contract cwd must be '.'")
    try:
        _normalize_argv(value.get("argv"))
    except (AttestationError, TypeError) as exc:
        errors.append(str(exc))
    for field in ("entrypoint", "output_dir"):
        try:
            normalized = _normalize_relative_path(str(value.get(field)))
        except AttestationError as exc:
            errors.append(str(exc))
        else:
            if normalized != value.get(field):
                errors.append(f"contract {field} is not normalized")
    try:
        if _normalize_resource_limits(
            value.get("resource_limits")
        ) != value.get("resource_limits"):
            errors.append("contract resource_limits are not normalized")
    except (AttestationError, TypeError) as exc:
        errors.append(str(exc))
    inputs = value.get("input_hashes")
    if not isinstance(inputs, Mapping):
        errors.append("contract input_hashes must be an object")
    else:
        errors.extend(_exact_keys(inputs, _INPUT_HASH_KEYS, "input_hashes"))
        for name in sorted(_INPUT_HASH_KEYS):
            record = inputs.get(name)
            if not isinstance(record, Mapping):
                errors.append(f"input_hashes.{name} must be an object")
                continue
            errors.extend(
                _exact_keys(
                    record,
                    _FILE_RECORD_KEYS,
                    f"input_hashes.{name}",
                )
            )
            _validate_digest_record(
                record,
                errors,
                f"input_hashes.{name}",
            )
            try:
                normalized = _normalize_relative_path(
                    str(record.get("path"))
                )
            except AttestationError as exc:
                errors.append(str(exc))
            else:
                if normalized != record.get("path"):
                    errors.append(
                        f"input_hashes.{name}.path is not normalized"
                    )
    try:
        if _normalize_env_keys(
            value.get("allowed_env_keys")
        ) != value.get("allowed_env_keys"):
            errors.append("contract allowed_env_keys are not normalized")
    except (AttestationError, TypeError) as exc:
        errors.append(str(exc))
    try:
        normalized_excludes = list(
            _normalize_excludes(value.get("artifact_excludes"))
        )
        if normalized_excludes != value.get("artifact_excludes"):
            errors.append("contract artifact_excludes are not normalized")
    except (AttestationError, TypeError) as exc:
        errors.append(str(exc))
    return errors


def _attestation_schema_errors(value: Mapping[str, Any]) -> list[str]:
    errors = _exact_keys(value, _ATTESTATION_KEYS, "attestation")
    if value.get("schema") != ATTESTATION_SCHEMA:
        errors.append("invalid attestation schema")
    if value.get("version") != SCHEMA_VERSION:
        errors.append("invalid attestation version")
    for field in ("idea_id", "job_id", "attempt_id", "mode"):
        if not isinstance(value.get(field), str) or not value.get(field):
            errors.append(f"invalid attestation {field}")
    _validate_sha256(value.get("contract_hash"), errors, "contract_hash")
    execution = value.get("execution")
    if not isinstance(execution, Mapping):
        errors.append("attestation execution must be an object")
    else:
        errors.extend(_exact_keys(execution, _EXECUTION_KEYS, "execution"))
        for field in ("started_at", "ended_at"):
            try:
                _normalize_timestamp(execution.get(field), field)
            except (AttestationError, TypeError) as exc:
                errors.append(str(exc))
        try:
            _strict_int(execution.get("returncode"), "returncode")
        except AttestationError as exc:
            errors.append(str(exc))
        try:
            _nonnegative_int(
                execution.get("allocated_gpus"),
                "allocated_gpus",
            )
        except AttestationError as exc:
            errors.append(str(exc))
    logs = value.get("logs")
    if not isinstance(logs, Mapping):
        errors.append("attestation logs must be an object")
    else:
        errors.extend(_exact_keys(logs, _LOG_KEYS, "logs"))
        for name in sorted(_LOG_KEYS):
            record = logs.get(name)
            if not isinstance(record, Mapping):
                errors.append(f"logs.{name} must be an object")
                continue
            errors.extend(
                _exact_keys(record, _LOG_RECORD_KEYS, f"logs.{name}")
            )
            _validate_digest_record(record, errors, f"logs.{name}")
    manifest = value.get("artifact_manifest")
    if not isinstance(manifest, Mapping):
        errors.append("attestation artifact_manifest must be an object")
    else:
        errors.extend(_manifest_schema_errors(manifest))
        expected_hash = canonical_json_sha256(manifest)
        if value.get("artifact_manifest_hash") != expected_hash:
            errors.append("artifact_manifest_hash does not match manifest")
    _validate_sha256(
        value.get("artifact_manifest_hash"),
        errors,
        "artifact_manifest_hash",
    )
    authentication = value.get("authentication")
    if not isinstance(authentication, Mapping):
        errors.append("attestation authentication must be an object")
    else:
        errors.extend(
            _exact_keys(authentication, _AUTH_KEYS, "authentication")
        )
        if authentication.get("algorithm") != "hmac-sha256":
            errors.append("invalid attestation authentication algorithm")
        if not isinstance(authentication.get("key_id"), str) or not (
            authentication.get("key_id")
        ):
            errors.append("invalid attestation key_id")
        _validate_sha256(
            authentication.get("signature"),
            errors,
            "authentication.signature",
        )
    return errors


def _validate_digest_record(
    record: Mapping[str, Any],
    errors: list[str],
    label: str,
) -> None:
    _validate_sha256(record.get("sha256"), errors, f"{label}.sha256")
    size = record.get("size_bytes")
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        errors.append(f"{label}.size_bytes must be a nonnegative integer")


def _validate_sha256(value: Any, errors: list[str], label: str) -> None:
    if not isinstance(value, str) or not _HEX_SHA256_RE.fullmatch(value):
        errors.append(f"{label} must be a lowercase SHA256 digest")


def _exact_keys(
    value: Mapping[str, Any],
    expected: set[str],
    label: str,
) -> list[str]:
    if not isinstance(value, Mapping):
        return [f"{label} must be an object"]
    actual = set(value)
    errors: list[str] = []
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        errors.append(f"{label} missing keys: {', '.join(missing)}")
    if unknown:
        errors.append(f"{label} has unknown keys: {', '.join(unknown)}")
    return errors


def _required_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise AttestationError(f"{label} must be a non-empty string")
    return value


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AttestationError(f"{label} must be an integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    result = _strict_int(value, label)
    if result < 0:
        raise AttestationError(f"{label} must be nonnegative")
    return result


def _normalize_timestamp(value: str | datetime, label: str) -> str:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value:
        parsed = _parse_timestamp(value)
    else:
        raise AttestationError(f"{label} must be an ISO-8601 timestamp")
    if parsed.tzinfo is None:
        raise AttestationError(f"{label} must include a timezone")
    return parsed.astimezone(UTC).isoformat(timespec="milliseconds")


def _parse_timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise AttestationError(
            "timestamp must be valid ISO-8601"
        ) from exc


def _require_signing_key(value: bytes) -> bytes:
    if not isinstance(value, bytes) or len(value) < 32:
        raise AttestationError(
            "signing_key must be at least 32 secret bytes"
        )
    return value


def _deep_copy_json(value: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise AttestationError(f"refusing to replace symlink: {path}")
    payload = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=parent,
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_json_object(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise AttestationError(f"refusing to read symlink: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AttestationError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AttestationError(f"JSON value must be an object: {path}")
    return value
