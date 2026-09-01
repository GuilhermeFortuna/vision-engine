from __future__ import annotations

from semantic_search.chunks import Chunk
from semantic_search.search import task_context


def test_task_context_prioritizes_docs(monkeypatch) -> None:
    chunks = [
        Chunk(
            path="docs/development/specs/VE-013-frame-messages-and-bounded-queues-spec.md",
            symbol="purpose",
            kind="doc",
            line_start=1,
            line_end=5,
            body="Frame messages and bounded queues",
            ve_id="VE-013",
            embedding=[1.0, 0.0],
        ),
        Chunk(
            path="src/pipeline/message.rs",
            symbol="DecodedFrame",
            kind="struct",
            line_start=10,
            line_end=20,
            body="pub struct DecodedFrame {}",
            module_path="pipeline::message",
            embedding=[0.5, 0.5],
        ),
    ]

    monkeypatch.setattr("semantic_search.search.get_index", lambda: chunks)
    monkeypatch.setattr("semantic_search.search.rank_chunks", lambda *args, **kwargs: [])
    results = task_context("VE-013", top_k=3)
    assert results[0]["kind"] == "doc"
    assert results[0]["ve_id"] == "VE-013"
