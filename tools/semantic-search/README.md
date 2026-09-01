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

### `semantic_code_search` parameters

- `query` — natural language or symbol-ish text
- `top_k` — number of results (default `8`)
- `path_prefix` — optional filter, e.g. `src/pipeline/`
- `kind` — optional filter: `fn`, `struct`, `enum`, `impl`, `mod`, `doc`

Results include `path`, `symbol`, `kind`, `line_start`, `line_end`, `score`, `match_reasons`, and `code`.

## Evaluation

```bash
cd tools/semantic-search
uv run pytest
uv run python eval/run_eval.py --reindex --phase 3
```

Golden queries live in [`eval/queries.json`](eval/queries.json). Add failed agent queries there to prevent regressions.

## When to reindex

Call `reindex` after:

- large refactors touching many `src/**/*.rs` files
- adding or renaming VE specs/plans
- first setup on a new machine

The index rebuilds automatically when indexed source files change.
