#!/usr/bin/env python3
"""Evaluate semantic search quality against golden queries."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from semantic_search.index import get_index, reindex
from semantic_search.rank import rank_chunks
from semantic_search.search import task_context

QUERIES_FILE = Path(__file__).resolve().parent / "queries.json"
TOP_K = 3
THRESHOLD = 0.8


def load_queries(phase: int | None) -> list[dict[str, object]]:
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    if phase is None:
        return queries
    return [query for query in queries if int(query["phase"]) <= phase]


def hit_in_top_k(
    ranked: list[dict[str, object]],
    expected: dict[str, object],
    top_k: int,
) -> bool:
    expected_path = str(expected.get("path", ""))
    expected_symbol = expected.get("symbol")
    for result in ranked[:top_k]:
        if expected_path and result["path"] != expected_path:
            continue
        if expected_symbol is None:
            return True
        if result.get("symbol") == expected_symbol or result.get("function") == expected_symbol:
            return True
    return False


def evaluate_query(entry: dict[str, object], top_k: int) -> tuple[bool, list[dict[str, object]]]:
    phase = int(entry["phase"])
    query = str(entry["query"])
    expected = entry["expected"]
    if not isinstance(expected, list):
        raise ValueError(f"invalid expected list for query: {query}")

    if phase >= 3 and query.upper().startswith("VE-"):
        ranked = task_context(query.split()[0], top_k=top_k)
    else:
        ranked = [
            {
                "path": item.chunk.path,
                "symbol": item.chunk.symbol,
                "function": item.chunk.symbol,
                "kind": item.chunk.kind,
                "score": item.score,
            }
            for item in rank_chunks(query, get_index(), top_k=top_k)
        ]

    # Each expected item must appear somewhere in top-k (not all in same slot).
    success = all(hit_in_top_k(ranked, item, top_k) for item in expected)
    return success, ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=None, help="Max phase to evaluate")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--reindex", action="store_true", help="Force index rebuild before eval")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    args = parser.parse_args()

    if args.reindex:
        reindex()
    else:
        get_index()

    queries = load_queries(args.phase)
    passed = 0
    for entry in queries:
        ok, ranked = evaluate_query(entry, args.top_k)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] phase={entry['phase']} query={entry['query']!r}")
        if not ok:
            print("  expected:", entry["expected"])
            print("  got:", [{"path": r["path"], "symbol": r.get("symbol")} for r in ranked[: args.top_k]])
        passed += int(ok)

    rate = passed / len(queries) if queries else 1.0
    print(f"\nTop-{args.top_k} hit rate: {passed}/{len(queries)} ({rate:.0%})")
    if rate < args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
