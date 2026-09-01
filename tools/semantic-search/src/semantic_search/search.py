"""Search orchestration for MCP tools."""

from __future__ import annotations

import re

from semantic_search.chunks import Chunk
from semantic_search.extract_plan_artifacts import PlanArtifacts
from semantic_search.index import get_index, get_plan_artifacts, index_response_meta
from semantic_search.rank import RankedChunk, rank_chunks

VE_ID_RE = re.compile(r"VE-\d{3}", re.IGNORECASE)

SCORE_FLOOR = 0.12
DEFAULT_TASK_TOP_K = 8

ANCHOR_SYMBOLS = (
    "Shutdown",
    "RunStats",
    "FrameMetrics",
    "QueueDepths",
    "StageTimings",
    "METRICS_AREA_BOTTOM",
    "DecodedFrame",
    "TrackedFrame",
)


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
    touchpoints: dict[str, list[str]] | None = None,
) -> dict[str, object]:
    response: dict[str, object] = {
        "results": results,
        "index": index_response_meta(
            index_load.meta,
            reindexed=index_load.reindexed,
            files_changed_since_index=index_load.files_changed_since_index,
        ),
    }
    if touchpoints is not None:
        response["touchpoints"] = touchpoints
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


def select_doc_headers(
    chunks: list[Chunk],
    normalized: str,
    artifacts: PlanArtifacts | None,
) -> list[Chunk]:
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

    selected: list[Chunk] = []
    if spec_docs:
        selected.append(spec_docs[0])
    if plan_docs:
        interfaces = next(
            (
                chunk
                for chunk in plan_docs
                if chunk.symbol == "interfaces_produced"
            ),
            None,
        )
        selected.append(interfaces or plan_docs[0])
    return selected[:2]


def best_chunk_for_path(chunks: list[Chunk], path: str) -> Chunk | None:
    candidates = [chunk for chunk in chunks if chunk.path == path]
    if not candidates:
        return None
    kind_priority = {
        "mod": 0,
        "struct": 1,
        "fn": 2,
        "const": 3,
        "impl": 4,
        "enum": 5,
        "script": 6,
        "plan_ref": 7,
    }
    return sorted(
        candidates,
        key=lambda chunk: (kind_priority.get(chunk.kind, 9), chunk.line_start),
    )[0]


def find_symbol_chunk(
    chunks: list[Chunk],
    symbol: str,
    preferred_paths: list[str],
) -> Chunk | None:
    for path in preferred_paths:
        for chunk in chunks:
            if (
                chunk.path == path
                and chunk.symbol == symbol
                and chunk.kind != "plan_ref"
            ):
                return chunk
    for chunk in chunks:
        if chunk.symbol == symbol and chunk.kind not in {"plan_ref", "doc"}:
            return chunk
    for chunk in chunks:
        if chunk.symbol == symbol:
            return chunk
    return None


def prioritize_symbols(
    symbols: list[str],
    chunks: list[Chunk],
    files: list[str],
) -> list[str]:
    filtered = [symbol for symbol in symbols if symbol not in {"BLOCKED", "Send"}]
    anchors = [symbol for symbol in ANCHOR_SYMBOLS if symbol in filtered]
    remaining = [symbol for symbol in filtered if symbol not in anchors]
    structs: list[str] = []
    others: list[str] = []
    for symbol in remaining:
        chunk = find_symbol_chunk(chunks, symbol, files)
        if chunk is not None and chunk.kind == "struct":
            structs.append(symbol)
        else:
            others.append(symbol)
    return anchors + structs + others


def task_context(ve_id: str, top_k: int = DEFAULT_TASK_TOP_K) -> dict[str, object]:
    """Return spec/plan sections and plan-named touchpoints for a VE task."""
    normalized = normalize_ve_id(ve_id)
    index_load = get_index()
    chunks = index_load.chunks
    artifacts_map = get_plan_artifacts()
    artifacts = artifacts_map.get(normalized)

    touchpoints = {
        "files": list(artifacts.files) if artifacts else [],
        "symbols": list(artifacts.symbols) if artifacts else [],
    }

    results: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()

    def add_chunk(
        chunk: Chunk,
        *,
        score: float,
        reasons: list[str],
    ) -> None:
        key = (chunk.path, chunk.symbol)
        if key in seen:
            return
        seen.add(key)
        results.append(
            chunk_to_result(
                RankedChunk(chunk=chunk, score=score, match_reasons=reasons)
            )
        )

    for doc in select_doc_headers(chunks, normalized, artifacts):
        add_chunk(doc, score=1.0, reasons=["task_doc"])

    if artifacts:
        ordered_symbols = prioritize_symbols(artifacts.symbols, chunks, artifacts.files)
        symbol_slots = max(3, top_k // 2)
        for symbol in ordered_symbols[:symbol_slots]:
            chunk = find_symbol_chunk(chunks, symbol, artifacts.files)
            if chunk is not None:
                add_chunk(chunk, score=0.9, reasons=[f"task_symbol:{symbol}"])

        for path in [item for item in artifacts.files if item.startswith("scripts/")]:
            chunk = best_chunk_for_path(chunks, path)
            if chunk is not None:
                add_chunk(chunk, score=0.93, reasons=[f"task_script:{path}"])

        for path in [item for item in artifacts.files if not item.startswith("scripts/")][:4]:
            preferred_symbol = next(
                (
                    symbol
                    for symbol in artifacts.symbols
                    if symbol not in {"BLOCKED", "Send"}
                    and find_symbol_chunk(chunks, symbol, [path]) is not None
                ),
                None,
            )
            chunk = (
                find_symbol_chunk(chunks, preferred_symbol, [path])
                if preferred_symbol
                else best_chunk_for_path(chunks, path)
            )
            if chunk is not None:
                key = (chunk.path, chunk.symbol)
                if key not in seen:
                    add_chunk(chunk, score=0.95, reasons=[f"task_file:{path}"])

        for symbol in ordered_symbols[symbol_slots:]:
            chunk = find_symbol_chunk(chunks, symbol, artifacts.files)
            if chunk is not None:
                add_chunk(chunk, score=0.88, reasons=[f"task_symbol:{symbol}"])

    remaining = max(0, top_k - len(results))
    if remaining > 0 and artifacts and artifacts.symbols:
        fill_query = " ".join(artifacts.symbols[:10])
        ranked = rank_chunks(
            fill_query,
            [chunk for chunk in chunks if chunk.kind not in {"doc", "plan_ref"}],
            top_k=remaining + 2,
            score_floor=SCORE_FLOOR,
        )
        for item in ranked:
            key = (item.chunk.path, item.chunk.symbol)
            if key in seen:
                continue
            item.match_reasons.insert(0, f"task:{normalized}")
            results.append(chunk_to_result(item))
            seen.add(key)
            if len(results) >= top_k:
                break

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
        touchpoints=touchpoints,
    )
