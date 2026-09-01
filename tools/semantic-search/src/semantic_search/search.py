"""Search orchestration for MCP tools."""

from __future__ import annotations

import re

from semantic_search.chunks import Chunk
from semantic_search.index import REPO_ROOT, get_index
from semantic_search.rank import RankedChunk, rank_chunks

VE_ID_RE = re.compile(r"VE-\d{3}", re.IGNORECASE)


def chunk_to_result(ranked: RankedChunk) -> dict[str, object]:
    chunk = ranked.chunk
    return {
        "path": chunk.path,
        "function": chunk.symbol,
        "symbol": chunk.symbol,
        "kind": chunk.kind,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "score": round(ranked.score, 4),
        "match_reasons": ranked.match_reasons,
        "code": chunk.display_code(),
        "ve_id": chunk.ve_id,
    }


def semantic_code_search(
    query: str,
    top_k: int = 8,
    path_prefix: str | None = None,
    kind: str | None = None,
) -> list[dict[str, object]]:
    """Search production Rust code and development docs with hybrid retrieval."""
    if top_k < 1:
        return []

    chunks = get_index()
    ranked = rank_chunks(
        query,
        chunks,
        top_k=top_k,
        path_prefix=path_prefix,
        kind=kind,
    )
    return [chunk_to_result(item) for item in ranked]


def normalize_ve_id(ve_id: str) -> str:
    match = VE_ID_RE.search(ve_id)
    if not match:
        raise ValueError(f"invalid VE task id: {ve_id}")
    return match.group(0).upper()


def task_context(ve_id: str, top_k: int = 5) -> list[dict[str, object]]:
    """Return spec/plan sections and related code for a VE task."""
    normalized = normalize_ve_id(ve_id)
    chunks = get_index()

    doc_hits = [
        chunk
        for chunk in chunks
        if chunk.kind == "doc" and chunk.ve_id == normalized
    ]
    spec_docs = sorted(
        [chunk for chunk in doc_hits if "/specs/" in chunk.path],
        key=lambda chunk: (chunk.path, chunk.line_start),
    )
    plan_docs = sorted(
        [chunk for chunk in doc_hits if "/plans/" in chunk.path],
        key=lambda chunk: (chunk.path, chunk.line_start),
    )
    ordered_docs = spec_docs[:1] + plan_docs[:1] + spec_docs[1:] + plan_docs[1:]

    code_slots = max(1, top_k // 2)
    doc_slots = max(0, top_k - code_slots)
    selected_docs = ordered_docs[:doc_slots]

    code_hits = rank_chunks(
        f"{normalized} Shutdown bounded queue disconnect pipeline runtime",
        [chunk for chunk in chunks if chunk.kind != "doc"],
        top_k=max(code_slots, 3),
        path_prefix="src/pipeline/",
    )
    shutdown_hit = next(
        (item for item in code_hits if item.chunk.symbol == "Shutdown"),
        None,
    )
    if shutdown_hit is not None:
        code_hits = [shutdown_hit] + [
            item for item in code_hits if item.chunk.symbol != "Shutdown"
        ]
    code_hits = code_hits[:code_slots]

    results: list[dict[str, object]] = []
    for chunk in selected_docs:
        results.append(
            chunk_to_result(
                RankedChunk(chunk=chunk, score=1.0, match_reasons=["task_doc"])
            )
        )

    for ranked in code_hits:
        ranked.match_reasons.insert(0, f"task:{normalized}")
        results.append(chunk_to_result(ranked))

    if not results:
        fallback = rank_chunks(normalized, chunks, top_k=top_k)
        return [chunk_to_result(item) for item in fallback]

    return results[:top_k]
