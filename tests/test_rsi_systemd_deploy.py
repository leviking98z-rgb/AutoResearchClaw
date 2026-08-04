from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEMD_DIR = REPO_ROOT / "deploy" / "systemd"
INSTALLER = SYSTEMD_DIR / "autoresearch-rsi-install"
INSTANCE = "test-rsi-campaign"


def _load_installer():
    loader = SourceFileLoader("rsi_systemd_installer", str(INSTALLER))
    spec = importlib.util.spec_from_loader("rsi_systemd_installer", loader)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_file_contains_no_secret_value() -> None:
    installer = _load_installer()
    text = installer.build_environment(
        campaign_dir=Path("/campaign"),
        campaign={"campaign_id": INSTANCE},
        repo_root=REPO_ROOT,
        owner="owner-id",
        nodes="node-a,node-b",
        purpose="RSI pool",
        pool_config=REPO_ROOT / "config.cluster32.yaml",
        lease_heartbeat=Path("/state/lease-heartbeat.json"),
    )
    assert "BRIDGE_LOCAL_API_KEY" not in text
    assert "owner-id" in text
    assert "RSI_CAMPAIGN_DIR=\"/campaign\"" in text
    assert 'AUTORESEARCH_CLAIM_OWNER="owner-id"' in text
    assert 'CB_SID="owner-id"' in text


def test_installer_stages_units_without_starting_systemd(tmp_path: Path) -> None:
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "campaign.json").write_text(
        json.dumps({"campaign_id": INSTANCE}),
        encoding="utf-8",
    )
    unit_dir = tmp_path / "units"
    env_dir = tmp_path / "env"
    completed = subprocess.run(
        [
            sys.executable,
            str(INSTALLER),
            "--campaign",
            str(campaign),
            "--repo-root",
            str(REPO_ROOT),
            "--owner",
            "owner-id",
            "--unit-dir",
            str(unit_dir),
            "--env-dir",
            str(env_dir),
            "--no-daemon-reload",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["started"] is False
    assert result["enabled"] is False
    assert result["daemon_reloaded"] is False
    assert (unit_dir / "autoresearch-rsi-supervisor@.service").is_file()
    assert (unit_dir / "autoresearch-rsi-alert@.service").is_file()
    env_file = env_dir / f"{INSTANCE}.env"
    assert env_file.is_file()
    assert env_file.stat().st_mode & 0o777 == 0o600
    alert_env = env_dir / "alerts.env"
    assert alert_env.is_file()
    assert alert_env.stat().st_mode & 0o777 == 0o600


def test_guard_rejects_duplicate_supervisor(tmp_path: Path) -> None:
    campaign = tmp_path / INSTANCE
    campaign.mkdir()
    (campaign / "campaign.json").write_text(
        json.dumps({"campaign_id": INSTANCE}),
        encoding="utf-8",
    )
    fields = Path(f"/proc/{os.getpid()}/stat").read_text(encoding="utf-8").split()
    (campaign / "state.json").write_text(
        json.dumps(
            {
                "pid": os.getpid(),
                "supervisor_start_ticks": int(fields[21]),
            }
        ),
        encoding="utf-8",
    )
    env = os.environ.copy()
    env["RSI_CAMPAIGN_DIR"] = str(campaign)
    env["RSI_REPO_ROOT"] = str(REPO_ROOT)
    completed = subprocess.run(
        [
            sys.executable,
            str(SYSTEMD_DIR / "autoresearch-rsi-guard"),
            "start-supervisor",
            INSTANCE,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode != 0
    assert "already running" in completed.stderr


def test_service_templates_use_boot_safe_restart_semantics() -> None:
    supervisor = (
        SYSTEMD_DIR / "autoresearch-rsi-supervisor@.service"
    ).read_text(encoding="utf-8")
    monitor = (
        SYSTEMD_DIR / "autoresearch-rsi-monitor@.service"
    ).read_text(encoding="utf-8")
    keepalive = (
        SYSTEMD_DIR / "autoresearch-rsi-keepalive@.service"
    ).read_text(encoding="utf-8")

    assert "Restart=on-failure" in supervisor
    assert "OnFailure=autoresearch-rsi-alert@%N.service" in supervisor
    assert "ExecStop=" not in supervisor
    assert "StandardOutput=journal" in supervisor
    assert "RestartPreventExitStatus=143" in supervisor
    assert "Restart=always" in monitor
    assert "OnFailure=autoresearch-rsi-alert@%N.service" in monitor
    assert "StandardOutput=journal" in monitor
    assert "RestartPreventExitStatus=143" in monitor
    assert "Restart=always" in keepalive
    assert "OnFailure=autoresearch-rsi-alert@%N.service" in keepalive
    assert "StandardOutput=journal" in keepalive
    assert "RestartPreventExitStatus=143" in keepalive


def test_supervisor_wrapper_auto_installs_runtime_dependencies() -> None:
    wrapper = (
        SYSTEMD_DIR / "autoresearch-rsi-supervisor-run"
    ).read_text(encoding="utf-8")

    ensure_command = (
        '"$REPO/bin/researchclaw-ensure-deps" --python "$PYTHON"'
    )
    resume_command = (
        'exec "$PYTHON" "$REPO/bin/rsi-resume" '
        '"$CAMPAIGN_DIR" --foreground'
    )
    assert ensure_command in wrapper
    assert resume_command in wrapper
    assert wrapper.index(ensure_command) < wrapper.index(resume_command)
