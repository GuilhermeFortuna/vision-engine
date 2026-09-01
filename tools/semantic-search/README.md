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
| `task_context` | VE spec/plan sections plus related code for a task id |
| `reindex` | Force rebuild of `.cache/index.json` |

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
  "index": {
    "built_at": "2026-09-01T21:10:00+00:00",
    "chunk_count": 142,
    "files_changed_since_index": 0,
    "reindexed": false
  }
}
```

When a query returns no hits, a `hint` field may suggest `path_prefix` or grep.

Check `index.reindexed` and `index.files_changed_since_index` during active implementation — a `true` / non-zero value means the index just refreshed from disk changes.

### `semantic_code_search` parameters

- `query` — natural language or symbol-ish text
- `top_k` — number of results (default `8`)
- `path_prefix` — optional filter, e.g. `src/pipeline/` (use during implementation to reduce noise)
- `kind` — optional filter: `fn`, `struct`, `enum`, `impl`, `mod`, `doc`

### `task_context` parameters

- `ve_id` — VE task id, e.g. `VE-014` (preferred parameter name)
- `task_id` — alias for `ve_id`
- `top_k` — number of results (default `5`)

Example: `task_context(ve_id="VE-014")` or `task_context(task_id="VE-014")`.

## Evaluation

```bash
cd tools/semantic-search
uv run pytest
uv run python eval/run_eval.py --reindex --phase 4
```

Golden queries live in [`eval/queries.json`](eval/queries.json). Phase 4 covers VE-014 runtime/shutdown/parity queries. Add failed agent queries there to prevent regressions.

## Index freshness

The index rebuilds **automatically on each search** when tracked source files change (mtime drift). That may take several seconds because embeddings call Ollama.

Call `reindex` explicitly after:

- first setup on a new machine
- troubleshooting when results still look wrong after a refresh

Tracked paths: `src/**/*.rs`, `docs/development/specs/VE-*.md`, `docs/development/plans/VE-*.md`, `AGENTS.md`, `PROJECT.md`.

## When to use vs grep/read

Use semantic search for orientation: architecture questions, VE task onboarding, and “where does X happen?” early in a task.

Once symbols or paths are known (`bounded`, `DecodeStage::next`, `runtime.rs`), grep and direct file reads are faster and authoritative.
