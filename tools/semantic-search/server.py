"""MCP server entrypoint."""

from __future__ import annotations

from mcp.server.mcpserver import MCPServer

from semantic_search.index import reindex as rebuild_index
from semantic_search.search import semantic_code_search as search_code
from semantic_search.search import task_context as search_task_context

mcp = MCPServer("semantic-search")


@mcp.tool()
def semantic_code_search(
    query: str,
    top_k: int = 8,
    path_prefix: str | None = None,
    kind: str | None = None,
) -> list[dict[str, object]]:
    """Search production Rust code and development docs with hybrid retrieval."""
    return search_code(
        query=query,
        top_k=top_k,
        path_prefix=path_prefix,
        kind=kind,
    )


@mcp.tool()
def reindex() -> dict[str, object]:
    """Force a full rebuild of the semantic search index."""
    return rebuild_index()


@mcp.tool()
def task_context(ve_id: str, top_k: int = 5) -> list[dict[str, object]]:
    """Return VE spec/plan sections and related code chunks for a task id."""
    return search_task_context(ve_id=ve_id, top_k=top_k)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
