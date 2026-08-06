from __future__ import annotations

import json
from pathlib import Path

import researchclaw.autoresearch_v2.research_memory as memory_module
from researchclaw.autoresearch_v2.config import ResearchMemoryConfig
from researchclaw.autoresearch_v2.ideas import candidate_to_idea
from researchclaw.autoresearch_v2.models import IdeaStatus
from researchclaw.autoresearch_v2.research_memory import InfoHubResearchMemory
from researchclaw.autoresearch_v2.store import V2Store


def _idea():
    idea = candidate_to_idea(
        {
            "id": "research-memory",
            "title": "Persistent RSI result",
            "family": "verifier",
            "research_question": "Does the gate help?",
            "falsifiable_hypothesis": "The gate improves accuracy.",
            "primary_metric": "accuracy",
            "datasets": ["GSM8K"],
            "models": ["Qwen/Qwen2.5-1.5B-Instruct"],
            "compute": {"gpu_count": 1, "wall_clock_hours": 1},
            "baselines": ["frozen policy"],
            "scores": {
                "novelty": 8,
                "scientific_importance": 8,
                "falsifiability": 8,
                "compute_tractability": 8,
                "reproducibility": 8,
                "meaningful_result_likelihood": 8,
                "risk": 2,
            },
        }
    )
    idea.status = IdeaStatus.COMPLETED_NEGATIVE
    idea.candidate["final_outcome"] = "informative_negative"
    return idea


def test_infohub_research_memory_upserts_one_stable_note(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    current = store.current_dir(idea.idea_id)
    (current / "artifacts" / "pilot").mkdir(parents=True)
    (current / "plan.json").write_text(
        json.dumps({"gate_statistic": {"name": "accuracy_gain"}}),
        encoding="utf-8",
    )
    (current / "artifacts" / "pilot" / "metrics.json").write_text(
        json.dumps(
            {
                "decision": "reject",
                "metrics": {"accuracy_gain": 0.0},
            }
        ),
        encoding="utf-8",
    )
    (current / "paper.md").write_text("# Negative result\n", encoding="utf-8")
    store.event(
        "job_finished",
        idea_id=idea.idea_id,
        decision="complete_negative",
        reason="threshold failed",
    )
    captured: dict[str, object] = {}

    class _Response:
        status = 200

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def read(self):
            return b'{"ok":true}'

    def urlopen(request, *, timeout):
        captured["url"] = request.full_url
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data)
        return _Response()

    monkeypatch.setattr(memory_module.urllib.request, "urlopen", urlopen)
    memory = InfoHubResearchMemory(
        config=ResearchMemoryConfig(
            url="http://infohub.test",
            timeout_sec=3,
        ),
        system_id="rsi-prod",
        store=store,
    )

    result = memory.reconcile(idea)

    assert result.ok is True
    assert result.external_id == (
        f"autoresearch-v2:rsi-prod:{idea.idea_id}"
    )
    assert captured["url"] == (
        "http://infohub.test/api/research-note/upsert"
    )
    body = captured["body"]
    assert body["kind"] == "research"
    assert body["status"] == "completed_negative"
    assert "informative_negative" in body["tags"]
    assert body["metadata"]["metrics"]["pilot"]["decision"] == "reject"
    artifact = next(
        item
        for item in body["metadata"]["artifacts"]
        if item["path"] == "paper.md"
    )
    assert artifact["kind"] == "paper"
    assert len(artifact["sha256"]) == 64
    assert "## Decision history" in body["content"]
    assert "# Negative result" in body["content"]


def test_infohub_failure_is_returned_not_raised(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()

    def urlopen(*args, **kwargs):
        del args, kwargs
        raise OSError("offline")

    monkeypatch.setattr(memory_module.urllib.request, "urlopen", urlopen)
    memory = InfoHubResearchMemory(
        config=ResearchMemoryConfig(),
        system_id="rsi-prod",
        store=store,
    )

    result = memory.reconcile(idea)

    assert result.ok is False
    assert "offline" in result.error


def test_infohub_fingerprint_changes_when_current_artifacts_change(
    tmp_path: Path,
) -> None:
    store = V2Store(tmp_path)
    store.initialize()
    idea = _idea()
    store.save_idea(idea)
    current = store.current_dir(idea.idea_id)
    current.mkdir(parents=True)
    paper = current / "paper.md"
    paper.write_text("# First\n", encoding="utf-8")
    memory = InfoHubResearchMemory(
        config=ResearchMemoryConfig(),
        system_id="rsi-prod",
        store=store,
    )

    first = memory.fingerprint(idea)
    paper.write_text("# Second, longer\n", encoding="utf-8")
    second = memory.fingerprint(idea)

    assert first != second
