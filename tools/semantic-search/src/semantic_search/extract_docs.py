"""Markdown documentation chunk extraction."""

from __future__ import annotations

import re
from pathlib import Path

from semantic_search.chunks import Chunk

VE_ID_RE = re.compile(r"(VE-\d{3})")
HEADING_RE = re.compile(r"^(#{2,3})\s+(.+)$")


def ve_id_from_path(rel_path: str) -> str | None:
    match = VE_ID_RE.search(Path(rel_path).name)
    return match.group(1) if match else None


def slugify(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.strip().lower())
    return slug.strip("_") or "section"


def chunk_markdown_file(repo_root: Path, rel_path: str) -> list[Chunk]:
    source = (repo_root / rel_path).read_text(encoding="utf-8")
    lines = source.splitlines()
    ve_id = ve_id_from_path(rel_path)
    module_path = rel_path.removesuffix(".md").replace("/", "::")
    chunks: list[Chunk] = []

    current_heading = Path(rel_path).stem
    current_level = 1
    section_start = 1
    section_lines: list[str] = []

    def flush(end_line: int) -> None:
        nonlocal section_lines, section_start, current_heading
        if not section_lines:
            return
        body = "\n".join(section_lines).strip()
        if not body:
            section_lines = []
            return
        symbol = slugify(current_heading)
        chunks.append(
            Chunk(
                path=rel_path,
                symbol=symbol,
                kind="doc",
                line_start=section_start,
                line_end=end_line,
                signature=current_heading,
                doc=current_heading,
                body=body,
                module_path=module_path,
                ve_id=ve_id,
            )
        )
        section_lines = []

    for index, line in enumerate(lines, start=1):
        heading = HEADING_RE.match(line)
        if heading:
            flush(index - 1)
            current_level = len(heading.group(1))
            current_heading = heading.group(2).strip()
            section_start = index
            section_lines = [line]
        else:
            section_lines.append(line)

    flush(len(lines))
    return chunks


def extract_doc_chunks(repo_root: Path) -> list[Chunk]:
    patterns = [
        "docs/development/specs/VE-*.md",
        "docs/development/plans/VE-*.md",
        "AGENTS.md",
        "PROJECT.md",
    ]
    chunks: list[Chunk] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(repo_root.glob(pattern)):
            rel_path = path.relative_to(repo_root).as_posix()
            if rel_path in seen:
                continue
            seen.add(rel_path)
            chunks.extend(chunk_markdown_file(repo_root, rel_path))
    return chunks
