"""Index build, persistence, and cache invalidation."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from semantic_search.chunks import Chunk
from semantic_search.embed import embed_texts
from semantic_search.extract_docs import extract_doc_chunks
from semantic_search.extract_rust import extract_rust_chunks

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
CACHE_FILE = CACHE_DIR / "index.json"
INDEX_VERSION = 2

SOURCE_PATTERNS = [
    "src/**/*.rs",
    "docs/development/specs/VE-*.md",
    "docs/development/plans/VE-*.md",
    "AGENTS.md",
    "PROJECT.md",
]

_INDEX: list[Chunk] | None = None


def tracked_files(repo_root: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in SOURCE_PATTERNS:
        for path in sorted(repo_root.glob(pattern)):
            if path not in seen:
                seen.add(path)
                files.append(path)
    return files


def source_mtimes(repo_root: Path) -> dict[str, float]:
    return {
        path.relative_to(repo_root).as_posix(): path.stat().st_mtime
        for path in tracked_files(repo_root)
    }


def cache_is_fresh(repo_root: Path, cached_mtimes: dict[str, float]) -> bool:
    current = source_mtimes(repo_root)
    if set(current) != set(cached_mtimes):
        return False
    return all(current[path] <= cached_mtimes[path] + 1e-6 for path in current)


def build_chunks(repo_root: Path) -> list[Chunk]:
    chunks = extract_rust_chunks(repo_root)
    chunks.extend(extract_doc_chunks(repo_root))
    if not chunks:
        return []

    texts = [chunk.embedding_text() for chunk in chunks]
    embeddings = embed_texts(texts)
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk.embedding = embedding
    return chunks


def save_index(repo_root: Path, chunks: list[Chunk]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": INDEX_VERSION,
        "built_at": datetime.now(UTC).isoformat(),
        "source_mtimes": source_mtimes(repo_root),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")


def load_index(repo_root: Path) -> list[Chunk] | None:
    if not CACHE_FILE.exists():
        return None
    payload = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    if payload.get("version") != INDEX_VERSION:
        return None
    cached_mtimes = payload.get("source_mtimes", {})
    if not isinstance(cached_mtimes, dict) or not cache_is_fresh(repo_root, cached_mtimes):
        return None
    raw_chunks = payload.get("chunks", [])
    if not isinstance(raw_chunks, list):
        return None
    return [Chunk.from_dict(item) for item in raw_chunks]


def build_index(repo_root: Path = REPO_ROOT) -> list[Chunk]:
    chunks = build_chunks(repo_root)
    save_index(repo_root, chunks)
    return chunks


def get_index(*, force: bool = False, repo_root: Path = REPO_ROOT) -> list[Chunk]:
    global _INDEX
    if force:
        _INDEX = build_index(repo_root)
        return _INDEX
    if _INDEX is not None:
        return _INDEX
    cached = load_index(repo_root)
    if cached is not None:
        _INDEX = cached
        return _INDEX
    _INDEX = build_index(repo_root)
    return _INDEX


def reindex() -> dict[str, object]:
    chunks = get_index(force=True)
    return {
        "status": "ok",
        "chunk_count": len(chunks),
        "cache_file": str(CACHE_FILE),
    }
