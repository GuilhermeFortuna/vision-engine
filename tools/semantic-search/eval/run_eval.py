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
from semantic_search.search import semantic_code_search, task_context

QUERIES_FILE = Path(__file__).resolve().parent / "queries.json"
TOP_K = 3
TASK_TOP_K = 8
THRESHOLD = 0.8


def load_queries(phase: int | None, *, task_only: bool) -> list[dict[str, object]]:
    queries = json.loads(QUERIES_FILE.read_text(encoding="utf-8"))
    if task_only:
        queries = [query for query in queries if int(query["phase"]) >= 3]
    elif phase is not None:
        queries = [query for query in queries if int(query["phase"]) <= phase]
    return queries


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


def evaluate_query(
    entry: dict[str, object],
    default_top_k: int,
) -> tuple[bool, list[dict[str, object]]]:
    phase = int(entry["phase"])
    query = str(entry["query"])
    expected = entry["expected"]
    top_k = int(entry.get("top_k", default_top_k))
    if not isinstance(expected, list):
        raise ValueError(f"invalid expected list for query: {query}")

    if phase >= 3 and query.upper().startswith("VE-"):
        response = task_context(query.split()[0], top_k=top_k)
        ranked = response["results"]
    else:
        path_prefix = entry.get("path_prefix")
        response = semantic_code_search(
            query,
            top_k=top_k,
            path_prefix=str(path_prefix) if path_prefix else None,
        )
        ranked = [
            {
                "path": item["path"],
                "symbol": item["symbol"],
                "function": item["symbol"],
                "kind": item["kind"],
                "score": item["score"],
            }
            for item in response["results"]
        ]

    success = all(hit_in_top_k(ranked, item, top_k) for item in expected)
    return success, ranked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, default=None, help="Max phase to evaluate")
    parser.add_argument("--top-k", type=int, default=None, help="Default top-k for queries")
    parser.add_argument("--reindex", action="store_true", help="Force index rebuild before eval")
    parser.add_argument("--threshold", type=float, default=THRESHOLD)
    parser.add_argument(
        "--task-only",
        action="store_true",
        help="Evaluate only task_context queries (phase >= 3)",
    )
    args = parser.parse_args()

    if args.reindex:
        reindex()
    else:
        get_index()

    queries = load_queries(args.phase, task_only=args.task_only)
    default_top_k = args.top_k if args.top_k is not None else (TASK_TOP_K if args.task_only else TOP_K)
    passed = 0
    for entry in queries:
        top_k = int(entry.get("top_k", default_top_k))
        ok, ranked = evaluate_query(entry, default_top_k)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] phase={entry['phase']} top_k={top_k} query={entry['query']!r}")
        if not ok:
            print("  expected:", entry["expected"])
            print("  got:", [{"path": r["path"], "symbol": r.get("symbol")} for r in ranked[:top_k]])
        passed += int(ok)

    rate = passed / len(queries) if queries else 1.0
    print(f"\nTop-{default_top_k} hit rate: {passed}/{len(queries)} ({rate:.0%})")
    if rate < args.threshold:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
