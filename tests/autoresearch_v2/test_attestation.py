from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from researchclaw.autoresearch_v2.attestation import (
    AttestationError,
    AttestationValidationError,
    build_tree_manifest,
    canonical_json_sha256,
    create_execution_attestation,
    create_execution_contract,
    load_execution_attestation,
    load_execution_contract,
    verify_execution_attestation,
    verify_execution_attestation_or_raise,
    verify_execution_contract,
    verify_sha256_manifest,
    write_execution_attestation,
    write_execution_contract,
)

_KEY = b"controller-only-test-key-material!" * 2


def _project(tmp_path: Path) -> dict[str, Path]:
    project = tmp_path / "candidate"
    artifacts = project / "artifacts" / "pilot"
    artifacts.mkdir(parents=True)
    (project / "main.py").write_text("print('real run')\n", encoding="utf-8")
    (project / "plan.json").write_text('{"phase":"pilot"}\n', encoding="utf-8")
    (project / "build.json").write_text(
        '{"commands":{"pilot":"python main.py"}}\n',
        encoding="utf-8",
    )
    (artifacts / "metrics.json").write_text(
        '{"result_valid":true}\n',
        encoding="utf-8",
    )
    (artifacts / "runtime_evidence.json").write_text(
        '{"model_loaded":true}\n',
        encoding="utf-8",
    )
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("completed\n", encoding="utf-8")
    stderr.write_text("", encoding="utf-8")
    return {
        "project": project,
        "artifacts": artifacts,
        "stdout": stdout,
        "stderr": stderr,
    }


def _contract(paths: dict[str, Path]) -> dict[str, object]:
    project = paths["project"]
    return create_execution_contract(
        idea_id="idea-1",
        job_id="job-1",
        attempt_id="attempt-1",
        mode="pilot",
        argv=["python", "main.py", "--mode", "pilot"],
        cwd=project,
        entrypoint="main.py",
        output_dir="artifacts/pilot",
        resource_limits={
            "max_gpus": 1,
            "timeout_sec": 7200,
        },
        plan_path=project / "plan.json",
        build_path=project / "build.json",
        allowed_env_keys=[
            "AUTORESEARCH_V2_ATTEMPT_ID",
            "AUTORESEARCH_V2_IDEA_ID",
            "AUTORESEARCH_V2_OUTPUT_DIR",
        ],
    )


def _contract_inputs(paths: dict[str, Path]) -> dict[str, object]:
    project = paths["project"]
    return {
        "idea_id": "idea-1",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "mode": "pilot",
        "argv": ["python", "main.py", "--mode", "pilot"],
        "cwd": project,
        "entrypoint": "main.py",
        "output_dir": "artifacts/pilot",
        "resource_limits": {
            "max_gpus": 1,
            "timeout_sec": 7200,
        },
        "plan_path": project / "plan.json",
        "build_path": project / "build.json",
        "allowed_env_keys": [
            "AUTORESEARCH_V2_ATTEMPT_ID",
            "AUTORESEARCH_V2_IDEA_ID",
            "AUTORESEARCH_V2_OUTPUT_DIR",
        ],
    }


def _attestation_inputs(paths: dict[str, Path]) -> dict[str, object]:
    return {
        "signing_key": _KEY,
        "key_id": "canary-controller",
        "started_at": "2026-08-06T01:02:03+00:00",
        "ended_at": "2026-08-06T01:03:04+00:00",
        "returncode": 0,
        "allocated_gpus": 1,
        "stdout_path": paths["stdout"],
        "stderr_path": paths["stderr"],
        "artifact_dir": paths["artifacts"],
    }


def test_recursive_manifest_is_stable_and_honors_safe_exclusions(
    tmp_path: Path,
) -> None:
    root = tmp_path / "artifacts"
    (root / "nested").mkdir(parents=True)
    (root / "metrics.json").write_text("metrics", encoding="utf-8")
    (root / "nested" / "result.txt").write_text("result", encoding="utf-8")
    (root / "execution_attestation.json").write_text(
        "model-forged",
        encoding="utf-8",
    )
    (root / "cache").mkdir()
    (root / "cache" / "ignored.bin").write_bytes(b"ignored")

    manifest = build_tree_manifest(
        root,
        exclude=("execution_attestation.json", "cache"),
    )

    assert [item["path"] for item in manifest["files"]] == [
        "metrics.json",
        "nested/result.txt",
    ]
    assert manifest["excluded_paths"] == [
        "cache",
        "execution_attestation.json",
    ]
    assert verify_sha256_manifest(root, manifest) == canonical_json_sha256(
        manifest
    )


def test_manifest_rejects_symlinks_and_path_escape(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    outside = tmp_path / "secret"
    outside.write_text("secret", encoding="utf-8")
    (root / "leak").symlink_to(outside)

    with pytest.raises(AttestationError, match="symlink"):
        build_tree_manifest(root)
    with pytest.raises(AttestationError, match="unsafe relative path"):
        build_tree_manifest(root, exclude=("../secret",))


def test_contract_binds_command_paths_inputs_resources_and_environment(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    contract = _contract(paths)
    contract_path = tmp_path / "trusted" / "execution_contract.json"

    written = write_execution_contract(contract_path, contract)

    assert contract["entrypoint"] == "main.py"
    assert contract["output_dir"] == "artifacts/pilot"
    assert contract["cwd"] == "."
    assert contract["input_hashes"]["plan"]["path"] == "plan.json"
    assert contract["input_hashes"]["build"]["path"] == "build.json"
    assert contract["allowed_env_keys"] == sorted(
        contract["allowed_env_keys"]
    )
    assert written == contract
    assert load_execution_contract(
        contract_path,
        **_contract_inputs(paths),
    ) == contract


def test_contract_verification_detects_input_or_claim_tampering(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    contract = _contract(paths)
    paths["project"].joinpath("plan.json").write_text(
        '{"phase":"scale"}\n',
        encoding="utf-8",
    )

    with pytest.raises(
        AttestationValidationError,
        match="does not match trusted controller inputs",
    ):
        verify_execution_contract(contract, **_contract_inputs(paths))

    forged = copy.deepcopy(contract)
    forged["allowed_env_keys"].append("MODEL_SUPPLIED_SECRET")
    with pytest.raises(AttestationValidationError):
        verify_execution_contract(forged, **_contract_inputs(paths))


def test_contract_rejects_entrypoint_and_output_path_escape(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    inputs = _contract_inputs(paths)
    inputs["entrypoint"] = "../outside.py"
    (tmp_path / "outside.py").write_text("pass\n", encoding="utf-8")

    with pytest.raises(AttestationError, match="escapes cwd"):
        create_execution_contract(**inputs)

    inputs = _contract_inputs(paths)
    inputs["output_dir"] = "../outside"
    with pytest.raises(AttestationError, match="escapes cwd"):
        create_execution_contract(**inputs)


def test_contract_rejects_symlinked_entrypoint(tmp_path: Path) -> None:
    paths = _project(tmp_path)
    project = paths["project"]
    (project / "linked.py").symlink_to(project / "main.py")
    inputs = _contract_inputs(paths)
    inputs["entrypoint"] = "linked.py"
    inputs["argv"] = ["python", "linked.py"]

    with pytest.raises(AttestationError, match="symlink path component"):
        create_execution_contract(**inputs)


def test_controller_attestation_round_trip_is_strictly_verified(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    contract = _contract(paths)
    contract_path = tmp_path / "trusted" / "execution_contract.json"
    write_execution_contract(contract_path, contract)
    attestation = create_execution_attestation(
        contract,
        **_attestation_inputs(paths),
    )
    attestation_path = tmp_path / "trusted" / "execution_attestation.json"

    attestation_hash = write_execution_attestation(
        attestation_path,
        attestation,
    )

    assert attestation["contract_hash"] == canonical_json_sha256(contract)
    assert attestation["execution"]["allocated_gpus"] == 1
    assert attestation["logs"]["stdout"]["size_bytes"] == len("completed\n")
    assert attestation_hash == canonical_json_sha256(attestation)
    assert load_execution_attestation(
        attestation_path,
        contract=contract,
        **_attestation_inputs(paths),
    ) == attestation
    assert verify_execution_attestation(
        contract_path,
        attestation_path,
        paths["project"],
        signing_key=_KEY,
        key_id="canary-controller",
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
        started_at="2026-08-06T01:02:03+00:00",
        ended_at="2026-08-06T01:03:04+00:00",
        returncode=0,
        allocated_gpus=1,
    ) == []


def test_integration_verifier_returns_errors_instead_of_trusting_claims(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    contract = _contract(paths)
    contract_path = tmp_path / "trusted" / "execution_contract.json"
    attestation_path = tmp_path / "trusted" / "execution_attestation.json"
    write_execution_contract(contract_path, contract)
    attestation = create_execution_attestation(
        contract,
        **_attestation_inputs(paths),
    )
    write_execution_attestation(attestation_path, attestation)
    paths["artifacts"].joinpath("metrics.json").write_text(
        '{"fabricated":true}\n',
        encoding="utf-8",
    )

    errors = verify_execution_attestation(
        contract_path,
        attestation_path,
        paths["project"],
        signing_key=_KEY,
        key_id="canary-controller",
        stdout_path=paths["stdout"],
        stderr_path=paths["stderr"],
    )

    assert errors
    assert any("trusted execution inputs" in error for error in errors)


def test_model_cannot_forge_attestation_without_controller_key(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    contract = _contract(paths)
    attestation = create_execution_attestation(
        contract,
        **_attestation_inputs(paths),
    )
    forged = copy.deepcopy(attestation)
    forged["execution"]["returncode"] = 9

    with pytest.raises(
        AttestationValidationError,
        match="signature is invalid",
    ):
        verify_execution_attestation_or_raise(
            forged,
            contract=contract,
            **_attestation_inputs(paths),
        )

    resigned_by_model = create_execution_attestation(
        contract,
        **{
            **_attestation_inputs(paths),
            "signing_key": b"model-controlled-key-material!!" * 2,
        },
    )
    with pytest.raises(
        AttestationValidationError,
        match="signature is invalid",
    ):
        verify_execution_attestation_or_raise(
            resigned_by_model,
            contract=contract,
            **_attestation_inputs(paths),
        )


def test_attestation_detects_artifact_and_log_tampering(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    contract = _contract(paths)
    attestation = create_execution_attestation(
        contract,
        **_attestation_inputs(paths),
    )

    paths["artifacts"].joinpath("metrics.json").write_text(
        '{"result_valid":false}\n',
        encoding="utf-8",
    )
    with pytest.raises(
        AttestationValidationError,
        match="does not match trusted execution inputs",
    ):
        verify_execution_attestation_or_raise(
            attestation,
            contract=contract,
            **_attestation_inputs(paths),
        )

    paths["artifacts"].joinpath("metrics.json").write_text(
        '{"result_valid":true}\n',
        encoding="utf-8",
    )
    paths["stdout"].write_text("forged log\n", encoding="utf-8")
    with pytest.raises(
        AttestationValidationError,
        match="does not match trusted execution inputs",
    ):
        verify_execution_attestation_or_raise(
            attestation,
            contract=contract,
            **_attestation_inputs(paths),
        )


def test_attestation_file_must_live_outside_model_artifacts(
    tmp_path: Path,
) -> None:
    paths = _project(tmp_path)
    contract = _contract(paths)
    attestation = create_execution_attestation(
        contract,
        **_attestation_inputs(paths),
    )
    forged_path = paths["artifacts"] / "execution_attestation.json"
    forged_path.write_text(json.dumps(attestation), encoding="utf-8")

    with pytest.raises(
        AttestationValidationError,
        match="must be stored outside artifact_dir",
    ):
        verify_execution_attestation_or_raise(
            attestation,
            contract=contract,
            attestation_path=forged_path,
            **_attestation_inputs(paths),
        )
