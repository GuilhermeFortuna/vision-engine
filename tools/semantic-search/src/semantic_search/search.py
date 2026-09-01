"""Search orchestration for MCP tools."""

from __future__ import annotations

import re

from semantic_search.chunks import Chunk
from semantic_search.index import get_index, index_response_meta
from semantic_search.rank import RankedChunk, rank_chunks

VE_ID_RE = re.compile(r"VE-\d{3}", re.IGNORECASE)

SCORE_FLOOR = 0.12


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


def wrap_search_response(
    results: list[dict[str, object]],
    *,
    index_load,
    query: str | None = None,
) -> dict[str, object]:
    response: dict[str, object] = {
        "results": results,
        "index": index_response_meta(
            index_load.meta,
            reindexed=index_load.reindexed,
            files_changed_since_index=index_load.files_changed_since_index,
        ),
    }
    if not results and index_load.files_changed_since_index > 0 and query:
        response["hint"] = (
            "Query returned no hits after index refresh. "
            "Try path_prefix (e.g. 'src/pipeline/') or grep for known symbols."
        )
    elif not results and query:
        response["hint"] = (
            "Query returned no hits. "
            "Try path_prefix (e.g. 'src/pipeline/') or grep for known symbols."
        )
    return response


def semantic_code_search(
    query: str,
    top_k: int = 8,
    path_prefix: str | None = None,
    kind: str | None = None,
) -> dict[str, object]:
    """Search production Rust code and development docs with hybrid retrieval."""
    index_load = get_index()
    if top_k < 1:
        return wrap_search_response([], index_load=index_load, query=query)

    ranked = rank_chunks(
        query,
        index_load.chunks,
        top_k=top_k,
        path_prefix=path_prefix,
        kind=kind,
        score_floor=SCORE_FLOOR,
    )
    results = [chunk_to_result(item) for item in ranked]
    return wrap_search_response(results, index_load=index_load, query=query)


def normalize_ve_id(ve_id: str) -> str:
    match = VE_ID_RE.search(ve_id)
    if not match:
        raise ValueError(f"invalid VE task id: {ve_id}")
    return match.group(0).upper()


def task_code_query(normalized: str) -> str:
    task_number = int(normalized.split("-")[1])
    base = f"{normalized} Shutdown bounded queue disconnect pipeline runtime"
    if task_number >= 14:
        base += (
            " runtime Pipeline spawn next_tracked request_shutdown join"
            " FaultConfig threaded stage threads cascade orchestration"
        )
    return base


def task_context(ve_id: str, top_k: int = 5) -> dict[str, object]:
    """Return spec/plan sections and related code for a VE task."""
    normalized = normalize_ve_id(ve_id)
    index_load = get_index()
    chunks = index_load.chunks

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
        task_code_query(normalized),
        [chunk for chunk in chunks if chunk.kind != "doc"],
        top_k=max(code_slots, 3),
        path_prefix="src/pipeline/",
        score_floor=SCORE_FLOOR,
    )
    shutdown_chunk = next(
        (
            chunk
            for chunk in chunks
            if chunk.path == "src/pipeline/queue.rs" and chunk.symbol == "Shutdown"
        ),
        None,
    )
    if shutdown_chunk is not None:
        shutdown_hit = RankedChunk(
            chunk=shutdown_chunk,
            score=1.0,
            match_reasons=["task:shutdown_anchor"],
        )
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
        fallback = rank_chunks(
            normalized,
            chunks,
            top_k=top_k,
            score_floor=SCORE_FLOOR,
        )
        results = [chunk_to_result(item) for item in fallback]

    return wrap_search_response(
        results[:top_k],
        index_load=index_load,
        query=normalized,
    )
