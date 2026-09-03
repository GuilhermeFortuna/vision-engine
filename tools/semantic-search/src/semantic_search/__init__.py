"""Hybrid semantic code search for Vision Engine."""

from semantic_search.search import semantic_code_search, task_context
from semantic_search.index import reindex

__all__ = ["semantic_code_search", "task_context", "reindex"]


def main() -> None:
    from semantic_search.server import main as run_server

    run_server()
