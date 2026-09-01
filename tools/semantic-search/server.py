"""MCP server entrypoint."""

from __future__ import annotations

from typing import Annotated

from mcp.server.mcpserver import MCPServer
from pydantic import Field

from semantic_search.index import reindex as rebuild_index
from semantic_search.search import semantic_code_search as search_code
from semantic_search.search import task_context as search_task_context

mcp = MCPServer("semantic-search")


@mcp.tool()
def semantic_code_search(
    query: Annotated[str, Field(description="Natural language or symbol-ish text to search for")],
    top_k: Annotated[int, Field(description="Maximum number of results to return", ge=1)] = 8,
    path_prefix: Annotated[
        str | None,
        Field(
            description=(
                "Optional path filter, e.g. 'src/pipeline/'. "
                "Use during implementation to reduce noise."
            )
        ),
    ] = None,
    kind: Annotated[
        str | None,
        Field(description="Optional kind filter: fn, struct, enum, impl, const, mod, script, doc, plan_ref"),
    ] = None,
) -> dict[str, object]:
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
def task_context(
    ve_id: Annotated[
        str | None,
        Field(description="VE task id, e.g. 'VE-014'"),
    ] = None,
    task_id: Annotated[
        str | None,
        Field(description="Alias for ve_id, e.g. 'VE-014'"),
    ] = None,
    top_k: Annotated[int, Field(description="Maximum number of results to return", ge=1)] = 8,
) -> dict[str, object]:
    """Return VE spec/plan sections and related code chunks for a task id."""
    resolved = ve_id or task_id
    if not resolved:
        raise ValueError("Provide ve_id or task_id (e.g. 'VE-014').")
    return search_task_context(ve_id=resolved, top_k=top_k)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
