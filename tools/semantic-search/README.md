# Semantic Search MCP

Hybrid code and documentation retrieval for Vision Engine agents.

## Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv)
- Ollama running locally with `qwen3-embedding:4b`

```bash
ollama pull qwen3-embedding:4b
```

## Setup

```bash
cd tools/semantic-search
uv sync --extra dev
```

Cursor loads the MCP from [`.cursor/mcp.json`](../../.cursor/mcp.json).

## Tools

| Tool | Purpose |
|------|---------|
| `semantic_code_search` | Hybrid semantic + BM25 + symbol search (`top_k=8` default) |
| `task_context` | VE spec/plan sections plus plan-named touchpoints for a task id |
| `reindex` | Force rebuild of `.cache/index.json` and `.cache/plan_artifacts.json` |

### Response shape

All search tools return a wrapped object:

```json
{
  "results": [
    {
      "path": "src/pipeline/runtime.rs",
      "symbol": "spawn",
      "kind": "fn",
      "line_start": 69,
      "line_end": 71,
      "score": 0.8421,
      "match_reasons": ["semantic", "symbol:spawn"],
      "code": "..."
    }
  ],
  "touchpoints": {
    "files": ["src/pipeline/metrics.rs", "scripts/sustained-run.sh"],
    "symbols": ["RunStats", "QueueDepths", "METRICS_AREA_BOTTOM"]
  },
  "index": {
    "built_at": "2026-09-01T21:10:00+00:00",
    "chunk_count": 142,
    "files_changed_since_index": 0,
    "reindexed": false
  }
}
```

`touchpoints` is present on `task_context` responses and lists files/symbols extracted from the VE spec and plan.

When a query returns no hits, a `hint` field may suggest `path_prefix` or grep.

Check `index.reindexed` and `index.files_changed_since_index` during active implementation — a `true` / non-zero value means the index just refreshed from disk changes.

### `semantic_code_search` parameters

- `query` — natural language or symbol-ish text
- `top_k` — number of results (default `8`)
- `path_prefix` — optional filter, e.g. `src/pipeline/` (use during implementation to reduce noise)
- `kind` — optional filter: `fn`, `struct`, `enum`, `impl`, `const`, `mod`, `script`, `doc`, `plan_ref`

### `task_context` parameters

- `ve_id` — VE task id, e.g. `VE-014` (preferred parameter name)
- `task_id` — alias for `ve_id`
- `top_k` — number of results (default `8`)

Example: `task_context(ve_id="VE-015")` or `task_context(task_id="VE-015")`.

## Evaluation

```bash
cd tools/semantic-search
uv run pytest
uv run python eval/run_eval.py --reindex --phase 4
uv run python eval/run_eval.py --reindex --phase 5 --top-k 8
```

Golden queries live in [`eval/queries.json`](eval/queries.json). Phase 4 covers VE-014 runtime/shutdown/parity queries. Phase 5 covers VE-015 instrumentation, const/script discovery, and task_context touchpoints. Add failed agent queries there to prevent regressions.

## Index freshness

The index rebuilds **automatically on each search** when tracked source files change (mtime drift). That may take several seconds because embeddings call Ollama.

Call `reindex` explicitly after:

- first setup on a new machine
- troubleshooting when results still look wrong after a refresh

Tracked paths: `src/**/*.rs`, `scripts/**/*.sh`, `docs/development/specs/VE-*.md`, `docs/development/plans/VE-*.md`, `AGENTS.md`, `PROJECT.md`.

## Chunk kinds

| Kind | Source |
|------|--------|
| `fn`, `struct`, `enum`, `impl`, `const`, `mod` | Production Rust under `src/` |
| `script` | Shell scripts under `scripts/` |
| `doc` | VE specs/plans, `AGENTS.md`, `PROJECT.md` |
| `plan_ref` | File/symbol touchpoints extracted from VE docs at index time |

## When to use vs grep/read

Use semantic search for orientation: architecture questions, VE task onboarding, and “where does X happen?” early in a task.

After `task_context`, grep `touchpoints.symbols` and `touchpoints.files` before running more searches.

Once symbols or paths are known (`bounded`, `DecodeStage::next`, `runtime.rs`), grep and direct file reads are faster and authoritative.
