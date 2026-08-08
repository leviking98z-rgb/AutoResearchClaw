from __future__ import annotations

import json

import researchclaw.research_queue.research_memory as memory_module
from researchclaw.research_queue.config import ResearchMemoryConfig
from researchclaw.research_queue.models import (
    Conclusion,
    IdeaProposal,
    IdeaRecord,
    IdeaStatus,
)
from researchclaw.research_queue.research_memory import InfoHubResearchMemory
from researchclaw.research_queue.store import ResearchQueueStore


def test_queue_infohub_memory_upserts_complete_artifacts(
    tmp_path,
    monkeypatch,
) -> None:
    store = ResearchQueueStore(tmp_path)
    store.initialize()
    idea = IdeaRecord.from_proposal(
        IdeaProposal(
            title="Queue memory",
            question="Does it work?",
            hypothesis="It works.",
            treatment="A",
            control="B",
            primary_metric="ECE",
        )
    )
    idea.status = IdeaStatus.CONCLUDED
    idea.conclusion = Conclusion.NEGATIVE
    store.upsert_idea(idea)
    idea_dir = store.idea_dir(idea.idea_id)
    (idea_dir / "research_note.md").write_text("# Complete note\n")
    (idea_dir / "final_review.json").write_text(
        json.dumps({"scientific_valid": True, "hypothesis_supported": False})
    )
    captured = {}

    class Response:
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
        return Response()

    monkeypatch.setattr(memory_module.urllib.request, "urlopen", urlopen)
    memory = InfoHubResearchMemory(
        config=ResearchMemoryConfig(
            enabled=True,
            url="http://infohub.test",
            timeout_sec=3,
        ),
        system_id="queue-test",
        store=store,
    )

    result = memory.reconcile(idea)

    assert result.ok
    assert captured["url"] == "http://infohub.test/api/research-note/upsert"
    assert captured["body"]["content"] == "# Complete note\n"
    assert captured["body"]["metadata"]["final_review"]["scientific_valid"] is True
    assert any(
        item["path"] == "final_review.json"
        for item in captured["body"]["metadata"]["artifacts"]
    )
