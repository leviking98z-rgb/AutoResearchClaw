from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from researchclaw.factory import actor
from researchclaw.factory.actor import PipelineIdeaWorker
from researchclaw.factory.config import FactoryConfig
from researchclaw.factory.models import Idea, WorkItem, WorkKind


def _proc_stat(pid: int, *, state: str, start_ticks: int) -> str:
    fields_4_through_21 = ["0"] * 18
    trailing = [state, *fields_4_through_21, str(start_ticks)]
    return f"{pid} (worker process) {' '.join(trailing)}\n"


def _mock_proc_stat(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pid: int,
    state: str,
    start_ticks: int,
) -> None:
    real_read_text = Path.read_text
    stat_path = Path(f"/proc/{pid}/stat")

    def fake_read_text(
        path: Path,
        *args: object,
        **kwargs: object,
    ) -> str:
        if path == stat_path:
            return _proc_stat(
                pid,
                state=state,
                start_ticks=start_ticks,
            )
        return real_read_text(path, *args, **kwargs)

    monkeypatch.setattr(actor.sys, "platform", "linux")
    monkeypatch.setattr(actor.os, "kill", lambda _pid, _signal: None)
    monkeypatch.setattr(Path, "read_text", fake_read_text)


def test_pid_matches_live_process_and_rejects_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = 43210
    _mock_proc_stat(
        monkeypatch,
        pid=pid,
        state="S",
        start_ticks=12345,
    )

    assert actor._pid_matches(pid, 12345)
    assert not actor._pid_matches(pid, 54321)
    assert actor._start_ticks(pid) == 12345


@pytest.mark.parametrize("state", ["Z", "X", "x"])
def test_pid_matches_rejects_terminal_proc_states(
    monkeypatch: pytest.MonkeyPatch,
    state: str,
) -> None:
    pid = 43211
    _mock_proc_stat(
        monkeypatch,
        pid=pid,
        state=state,
        start_ticks=12345,
    )

    assert not actor._pid_matches(pid, 12345)
    assert not actor._pid_matches(pid, None)


def test_probe_marks_zombie_worker_finished(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pid = 43212
    start_ticks = 12345
    _mock_proc_stat(
        monkeypatch,
        pid=pid,
        state="Z",
        start_ticks=start_ticks,
    )
    worker = PipelineIdeaWorker(FactoryConfig())
    item = WorkItem(
        item_id="idea-screen",
        idea_id="idea",
        kind=WorkKind.PIPELINE,
        profile="screen",
        attempt=1,
    )
    idea_dir = tmp_path / "idea"
    worker_dir = worker._worker_dir(idea_dir, item)
    worker_dir.mkdir(parents=True)
    state_path = worker_dir / "state.json"
    state_path.write_text(
        json.dumps(
            {
                "state": "running",
                "pid": pid,
                "start_ticks": start_ticks,
                "started_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )

    probe = worker.probe(item=item, idea_dir=idea_dir)

    assert probe.state == "finished"
    assert probe.returncode == -1
    assert probe.pid == pid
    persisted = json.loads(state_path.read_text(encoding="utf-8"))
    assert persisted["state"] == "finished"
    assert persisted["returncode"] == -1


def test_pipeline_worker_propagates_complete_correlation_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        """project: {name: test, mode: full-auto}
research: {topic: stale topic}
runtime: {timezone: UTC}
notifications: {channel: stdout, target: ''}
knowledge_base: {backend: markdown, root: ''}
llm:
  provider: openai-compatible
  base_url: http://example.invalid
  api_key_env: TEST_KEY
""",
        encoding="utf-8",
    )
    config = FactoryConfig.from_mapping(
        {
            "factory": {
                "factory_id": "factory-a",
                "worker": {"pipeline_config": str(base)},
            }
        }
    )
    idea = Idea(
        idea_id="idea-a",
        title="RSI acceptance-gate study",
        research_question="Do calibrated gates prevent collapse?",
        falsifiable_hypothesis="Calibrated gates reduce false accepts.",
        primary_metric="false_accept_rate",
    )
    item = WorkItem(
        item_id="idea-a-screen",
        idea_id=idea.idea_id,
        kind=WorkKind.PIPELINE,
        profile="screen",
        attempt=2,
    )
    captured: dict[str, object] = {}

    def fake_popen(argv, **kwargs):
        captured["argv"] = argv
        captured["env"] = kwargs["env"]
        return SimpleNamespace(pid=43213)

    monkeypatch.setattr(actor.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(actor, "_start_ticks", lambda _pid: 12345)

    PipelineIdeaWorker(config).start(
        idea=idea,
        item=item,
        idea_dir=tmp_path / "idea",
    )

    child_env = captured["env"]
    assert isinstance(child_env, dict)
    assert child_env["RESEARCHCLAW_FACTORY_ID"] == "factory-a"
    assert child_env["RESEARCHCLAW_IDEA_ID"] == "idea-a"
    assert child_env["RESEARCHCLAW_WORK_ITEM_ID"] == "idea-a-screen"
    assert child_env["RESEARCHCLAW_WORK_ITEM_ATTEMPT"] == "2"
    events = [
        json.loads(line)
        for line in (
            tmp_path / "idea" / "operational_events.jsonl"
        ).read_text(encoding="utf-8").splitlines()
    ]
    assert events[-1]["factory_id"] == "factory-a"
