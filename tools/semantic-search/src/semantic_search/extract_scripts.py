"""Shell script chunk extraction."""

from __future__ import annotations

from pathlib import Path

from semantic_search.chunks import Chunk


def extract_script_chunks(repo_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for script_file in sorted(repo_root.glob("scripts/**/*.sh")):
        rel_path = script_file.relative_to(repo_root).as_posix()
        source = script_file.read_text(encoding="utf-8")
        stem = script_file.stem
        line_count = max(1, source.count("\n") + 1)
        chunks.append(
            Chunk(
                path=rel_path,
                symbol=stem,
                kind="script",
                line_start=1,
                line_end=line_count,
                signature=f"#!/bin/bash {stem}",
                doc="",
                body=source,
                module_path=f"scripts::{stem}",
            )
        )
    return chunks
