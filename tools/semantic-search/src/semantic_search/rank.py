"""Hybrid ranking: semantic + BM25 + symbol overlap."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from semantic_search.chunks import Chunk
from semantic_search.embed import cosine_similarity, embed_text

SEMANTIC_WEIGHT = 0.45
BM25_WEIGHT = 0.35
SYMBOL_WEIGHT = 0.20

KIND_WEIGHTS = {
    "fn": 1.15,
    "struct": 0.95,
    "enum": 0.95,
    "impl": 0.9,
    "const": 1.1,
    "mod": 0.85,
    "script": 1.0,
    "plan_ref": 0.9,
    "doc": 0.75,
}

CODE_QUERY_HINTS = (
    "opencv",
    "mat",
    "struct",
    "enum",
    "impl",
    "ownership",
    "thread",
    "threaded",
    "runtime",
    "spawn",
    "join",
    "cascade",
    "orchestrat",
    "send",
    "recv",
    "bounded",
    "frame",
)

ORCHESTRATION_QUERY_TOKENS = frozenset(
    {
        "thread",
        "threaded",
        "stage",
        "cascade",
        "spawn",
        "join",
        "runtime",
        "orchestrat",
        "next",
        "pipeline",
        "serial",
        "loop",
    }
)

INSTRUMENTATION_QUERY_TOKENS = frozenset(
    {
        "overlay",
        "metrics",
        "instrument",
        "queue_depth",
        "queue",
        "depth",
        "runstats",
        "percentile",
        "summary",
        "throughput",
        "fps",
    }
)

PERIPHERAL_SYMBOLS = frozenset({"open", "new", "default"})


def kind_weight(chunk: Chunk, query: str) -> float:
    weight = KIND_WEIGHTS.get(chunk.kind, 1.0)
    lowered = query.lower()
    if chunk.kind == "doc":
        if any(token in lowered for token in ("ve-", "spec", "plan", "requirement", "milestone")):
            return 1.0
        if any(hint in lowered for hint in CODE_QUERY_HINTS):
            return 0.35
    if chunk.kind == "script" and "scripts/" in lowered:
        return 1.2
    return weight


def tokenize(text: str) -> list[str]:
    normalized = text.replace("::", " ")
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", normalized)
    normalized = re.sub(r"[_\-]+", " ", normalized)
    return [token.lower() for token in re.findall(r"[A-Za-z0-9]+", normalized) if token]


def expand_query_tokens(tokens: list[str]) -> set[str]:
    expanded = set(tokens)
    for token in tokens:
        if token.endswith("s") and len(token) > 3:
            expanded.add(token[:-1])
    return expanded


def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if maximum == minimum:
        return [1.0 if maximum > 0.0 else 0.0 for _ in scores]
    return [(score - minimum) / (maximum - minimum) for score in scores]


def body_overlap_score(query_tokens: set[str], chunk: Chunk) -> float:
    body_tokens = set(tokenize(chunk.search_text()))
    if not query_tokens:
        return 0.0
    overlap = query_tokens & body_tokens
    return len(overlap) / len(query_tokens)


def symbol_score(query_tokens: set[str], chunk: Chunk) -> float:
    if not query_tokens:
        return 0.0

    symbol_lower = chunk.symbol.lower()
    if symbol_lower in query_tokens:
        return 1.0

    if any(token in symbol_lower for token in query_tokens if len(token) > 3):
        return 0.85

    symbol_tokens = set(tokenize(chunk.symbol))
    path_tokens = set(tokenize(chunk.path.replace("/", " ").replace(".", " ")))
    module_tokens = set(tokenize(chunk.module_path.replace("::", " ")))
    candidate = symbol_tokens | path_tokens | module_tokens

    if not candidate:
        return 0.0

    overlap = query_tokens & candidate
    return len(overlap) / len(query_tokens)


def match_reasons(
    query_tokens: set[str],
    chunk: Chunk,
    semantic: float,
    bm25: float,
    symbol: float,
) -> list[str]:
    reasons: list[str] = []
    if semantic > 0.0:
        reasons.append("semantic")
    if bm25 > 0.0:
        matched = sorted(query_tokens & set(tokenize(chunk.search_text())))
        if matched:
            reasons.append("bm25:" + ",".join(matched[:3]))
        else:
            reasons.append("bm25")
    if symbol > 0.0:
        reasons.append(f"symbol:{chunk.symbol}")
    return reasons


@dataclass
class RankedChunk:
    chunk: Chunk
    score: float
    match_reasons: list[str]


def rank_chunks(
    query: str,
    chunks: list[Chunk],
    *,
    top_k: int,
    path_prefix: str | None = None,
    kind: str | None = None,
    query_embedding: list[float] | None = None,
    score_floor: float | None = None,
) -> list[RankedChunk]:
    if top_k < 1 or not chunks:
        return []

    filtered = chunks
    if path_prefix:
        filtered = [chunk for chunk in filtered if chunk.path.startswith(path_prefix)]
    if kind:
        filtered = [chunk for chunk in filtered if chunk.kind == kind]
    if not filtered:
        return []

    query_tokens = expand_query_tokens(tokenize(query))
    orchestration_query = bool(query_tokens & ORCHESTRATION_QUERY_TOKENS)
    instrumentation_query = bool(query_tokens & INSTRUMENTATION_QUERY_TOKENS)
    message_query = bool(query_tokens & {"message", "messages", "timings", "stamp"})
    corpus_tokens = [tokenize(chunk.search_text()) for chunk in filtered]
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(tokenize(query)).tolist()

    if query_embedding is None:
        query_embedding = embed_text(query)

    semantic_scores = [
        cosine_similarity(query_embedding, chunk.embedding or [])
        for chunk in filtered
    ]
    symbol_scores = [symbol_score(query_tokens, chunk) for chunk in filtered]

    norm_semantic = normalize_scores(semantic_scores)
    norm_bm25 = normalize_scores(bm25_scores)
    norm_symbol = normalize_scores(symbol_scores)

    ranked: list[RankedChunk] = []
    for index, chunk in enumerate(filtered):
        fused = (
            SEMANTIC_WEIGHT * norm_semantic[index]
            + BM25_WEIGHT * norm_bm25[index]
            + SYMBOL_WEIGHT * norm_symbol[index]
            + 0.1 * body_overlap_score(query_tokens, chunk)
        )
        fused *= kind_weight(chunk, query)

        if chunk.kind == "plan_ref" and any(hint in query.lower() for hint in CODE_QUERY_HINTS):
            fused *= 0.6
        if chunk.kind == "const" and chunk.symbol.lower() in query_tokens:
            fused *= 1.35
        if chunk.kind == "script" and path_prefix and path_prefix.startswith("scripts/"):
            fused *= 1.2

        if instrumentation_query:
            if chunk.path.endswith("metrics.rs"):
                fused *= 1.25
            if chunk.path.endswith("render.rs"):
                fused *= 1.15
            if chunk.symbol in {
                "draw_metrics_overlay",
                "log_playback_summary",
                "log_instrumentation_summary",
                "RunStats",
                "QueueDepths",
                "METRICS_AREA_BOTTOM",
            }:
                fused *= 1.2

        if orchestration_query and not message_query:
            if chunk.path.endswith("runtime.rs"):
                fused *= 1.35
            if chunk.kind == "fn" and chunk.symbol in {
                "spawn",
                "join",
                "next_tracked",
                "request_shutdown",
            }:
                fused *= 1.2
            if chunk.kind == "struct" and chunk.symbol in {"Pipeline", "FaultConfig"}:
                fused *= 1.1
            if chunk.kind == "struct" and chunk.symbol == "Pipeline" and "pipeline" in query_tokens:
                fused *= 1.35
        if chunk.path.endswith("queue.rs") and chunk.symbol in query_tokens:
            fused *= 1.2
        if chunk.symbol == "request" and {"send", "recv", "shutdown"} & query_tokens:
            fused *= 0.85
        if chunk.symbol == "Shutdown" and "shutdown" in query_tokens:
            fused *= 1.25
        if chunk.symbol == "StageTimings" and "timings" in query_tokens:
            fused *= 1.15
        if chunk.symbol == "DecodedFrame" and "mat" in query_tokens:
            fused *= 1.5
        if chunk.path.endswith("message.rs") and "mat" in query_tokens:
            fused *= 1.1

        if message_query and chunk.path.endswith("message.rs"):
            fused *= 1.2
        if message_query and chunk.symbol in {"StageTimings", "TrackedFrame", "DecodedFrame"}:
            fused *= 1.15

        if (
            len(query_tokens) >= 2
            and chunk.symbol in PERIPHERAL_SYMBOLS
            and chunk.symbol.lower() not in query_tokens
        ):
            fused *= 0.85

        reasons = match_reasons(
            query_tokens,
            chunk,
            norm_semantic[index],
            norm_bm25[index],
            norm_symbol[index],
        )
        ranked.append(RankedChunk(chunk=chunk, score=fused, match_reasons=reasons))

    ranked.sort(
        key=lambda item: (
            -item.score,
            {
                "fn": 0,
                "const": 1,
                "struct": 2,
                "enum": 3,
                "impl": 4,
                "mod": 5,
                "script": 6,
                "plan_ref": 7,
                "doc": 8,
            }.get(item.chunk.kind, 9),
            item.chunk.symbol,
        )
    )

    deduped: list[RankedChunk] = []
    seen: set[tuple[str, str]] = set()
    for item in ranked:
        if score_floor is not None and item.score < score_floor:
            continue
        key = (item.chunk.path, item.chunk.symbol)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
        if len(deduped) >= top_k:
            break

    return deduped
