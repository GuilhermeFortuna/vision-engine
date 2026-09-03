"""Index build, persistence, and cache invalidation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from semantic_search.chunks import Chunk
from semantic_search.embed import embed_texts
from semantic_search.extract_docs import extract_doc_chunks
from semantic_search.extract_plan_artifacts import (
    PlanArtifacts,
    build_plan_artifacts,
    load_plan_artifacts,
    plan_ref_chunks,
    save_plan_artifacts,
)
from semantic_search.extract_rust import extract_rust_chunks
from semantic_search.extract_scripts import extract_script_chunks

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
CACHE_FILE = CACHE_DIR / "index.json"
PLAN_ARTIFACTS_FILE = CACHE_DIR / "plan_artifacts.json"
INDEX_VERSION = 3

SOURCE_PATTERNS = [
    "src/**/*.rs",
    "scripts/**/*.sh",
    "docs/development/specs/VE-*.md",
    "docs/development/plans/VE-*.md",
    "AGENTS.md",
    "PROJECT.md",
]


@dataclass
class IndexMeta:
    built_at: str
    source_mtimes: dict[str, float]
    chunk_count: int


@dataclass
class IndexLoad:
    chunks: list[Chunk]
    meta: IndexMeta
    reindexed: bool
    files_changed_since_index: int


_INDEX: list[Chunk] | None = None
_INDEX_META: IndexMeta | None = None


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


def index_staleness(repo_root: Path, meta: IndexMeta) -> dict[str, object]:
    current = source_mtimes(repo_root)
    cached = meta.source_mtimes
    added = set(current) - set(cached)
    removed = set(cached) - set(current)
    changed = [
        path
        for path in set(current) & set(cached)
        if current[path] > cached[path] + 1e-6
    ]
    files_changed = len(changed)
    files_added = len(added)
    files_removed = len(removed)
    return {
        "files_changed": files_changed,
        "files_added": files_added,
        "files_removed": files_removed,
        "files_changed_since_index": files_changed + files_added + files_removed,
        "is_stale": not cache_is_fresh(repo_root, cached),
    }


def meta_from_payload(
    payload: dict[str, object], *, chunk_count: int | None = None
) -> IndexMeta:
    source_mtimes_raw = payload.get("source_mtimes", {})
    if not isinstance(source_mtimes_raw, dict):
        source_mtimes_raw = {}
    cached_mtimes = {str(key): float(value) for key, value in source_mtimes_raw.items()}
    if chunk_count is None:
        chunks = payload.get("chunks", [])
        chunk_count = len(chunks) if isinstance(chunks, list) else 0
    return IndexMeta(
        built_at=str(payload.get("built_at", "")),
        source_mtimes=cached_mtimes,
        chunk_count=chunk_count,
    )


def index_response_meta(
    meta: IndexMeta,
    *,
    reindexed: bool,
    files_changed_since_index: int,
) -> dict[str, object]:
    return {
        "built_at": meta.built_at,
        "chunk_count": meta.chunk_count,
        "files_changed_since_index": files_changed_since_index,
        "reindexed": reindexed,
    }


def build_chunks(repo_root: Path) -> list[Chunk]:
    chunks = extract_rust_chunks(repo_root)
    chunks.extend(extract_script_chunks(repo_root))
    chunks.extend(extract_doc_chunks(repo_root))
    artifacts = build_plan_artifacts(repo_root)
    save_plan_artifacts(artifacts)
    chunks.extend(plan_ref_chunks(artifacts))
    if not chunks:
        return []

    texts = [chunk.embedding_text() for chunk in chunks]
    embeddings = embed_texts(texts)
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk.embedding = embedding
    return chunks


def save_index(repo_root: Path, chunks: list[Chunk]) -> IndexMeta:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    mtimes = source_mtimes(repo_root)
    built_at = datetime.now(UTC).isoformat()
    payload = {
        "version": INDEX_VERSION,
        "built_at": built_at,
        "source_mtimes": mtimes,
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    CACHE_FILE.write_text(json.dumps(payload), encoding="utf-8")
    return IndexMeta(built_at=built_at, source_mtimes=mtimes, chunk_count=len(chunks))


def load_index(repo_root: Path) -> tuple[list[Chunk], IndexMeta] | None:
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
    chunks = [Chunk.from_dict(item) for item in raw_chunks]
    return chunks, meta_from_payload(payload, chunk_count=len(chunks))


def build_index(repo_root: Path = REPO_ROOT) -> tuple[list[Chunk], IndexMeta]:
    chunks = build_chunks(repo_root)
    meta = save_index(repo_root, chunks)
    return chunks, meta


def get_index(*, force: bool = False, repo_root: Path = REPO_ROOT) -> IndexLoad:
    global _INDEX, _INDEX_META

    if force:
        chunks, meta = build_index(repo_root)
        _INDEX = chunks
        _INDEX_META = meta
        return IndexLoad(
            chunks=chunks,
            meta=meta,
            reindexed=True,
            files_changed_since_index=0,
        )

    if _INDEX is not None and _INDEX_META is not None:
        if cache_is_fresh(repo_root, _INDEX_META.source_mtimes):
            return IndexLoad(
                chunks=_INDEX,
                meta=_INDEX_META,
                reindexed=False,
                files_changed_since_index=0,
            )
        staleness = index_staleness(repo_root, _INDEX_META)
        files_changed = int(staleness["files_changed_since_index"])
        chunks, meta = build_index(repo_root)
        _INDEX = chunks
        _INDEX_META = meta
        return IndexLoad(
            chunks=chunks,
            meta=meta,
            reindexed=True,
            files_changed_since_index=files_changed,
        )

    cached = load_index(repo_root)
    if cached is not None:
        chunks, meta = cached
        _INDEX = chunks
        _INDEX_META = meta
        return IndexLoad(
            chunks=chunks,
            meta=meta,
            reindexed=False,
            files_changed_since_index=0,
        )

    chunks, meta = build_index(repo_root)
    _INDEX = chunks
    _INDEX_META = meta
    return IndexLoad(
        chunks=chunks,
        meta=meta,
        reindexed=True,
        files_changed_since_index=0,
    )


def reindex() -> dict[str, object]:
    index_load = get_index(force=True)
    return {
        "status": "ok",
        "cache_file": str(CACHE_FILE),
        "plan_artifacts_file": str(PLAN_ARTIFACTS_FILE),
        "index": index_response_meta(
            index_load.meta,
            reindexed=True,
            files_changed_since_index=0,
        ),
    }


def get_plan_artifacts(repo_root: Path = REPO_ROOT) -> dict[str, PlanArtifacts]:
    artifacts = load_plan_artifacts()
    if artifacts:
        return artifacts
    artifacts = build_plan_artifacts(repo_root)
    save_plan_artifacts(artifacts)
    return artifacts
