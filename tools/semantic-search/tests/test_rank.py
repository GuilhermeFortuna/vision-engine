from __future__ import annotations

from semantic_search.chunks import Chunk
from semantic_search.rank import rank_chunks, tokenize


def test_tokenize_splits_camel_and_snake() -> None:
    tokens = tokenize("DecodeStage next bounded_queue")
    assert "decode" in tokens
    assert "stage" in tokens
    assert "next" in tokens
    assert "bounded" in tokens
    assert "queue" in tokens


def test_hybrid_rank_prefers_symbol_match() -> None:
    chunks = [
        Chunk(
            path="src/pipeline/queue.rs",
            symbol="send",
            kind="fn",
            line_start=1,
            line_end=2,
            body="pub fn send() {}",
            module_path="pipeline::queue",
            embedding=[0.5, 0.5],
        ),
        Chunk(
            path="src/pipeline/metrics.rs",
            symbol="record_frame",
            kind="fn",
            line_start=1,
            line_end=2,
            body="pub fn record_frame() {}",
            module_path="pipeline::metrics",
            embedding=[0.5, 0.5],
        ),
    ]
    ranked = rank_chunks(
        "queue send",
        chunks,
        top_k=1,
        query_embedding=[0.5, 0.5],
    )
    assert ranked[0].chunk.symbol == "send"
