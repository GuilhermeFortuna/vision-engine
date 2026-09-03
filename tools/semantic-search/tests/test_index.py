from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest

from semantic_search.chunks import Chunk
from semantic_search.index import (
    IndexMeta,
    cache_is_fresh,
    get_index,
    index_staleness,
    source_mtimes,
)


@dataclass
class FakeIndexState:
    build_count: int = 0


def make_chunk(path: str = "src/example.rs", symbol: str = "example") -> Chunk:
    return Chunk(
        path=path,
        symbol=symbol,
        kind="fn",
        line_start=1,
        line_end=5,
        body="pub fn example() {}",
        embedding=[1.0, 0.0],
    )


def install_fake_build(monkeypatch: pytest.MonkeyPatch, state: FakeIndexState) -> None:
    def fake_build(repo_root: Path):
        state.build_count += 1
        meta = IndexMeta(
            built_at=f"2026-01-01T00:00:0{state.build_count}+00:00",
            source_mtimes=source_mtimes(repo_root),
            chunk_count=1,
        )
        return [make_chunk()], meta

    monkeypatch.setattr("semantic_search.index.build_index", fake_build)


def reset_index_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("semantic_search.index._INDEX", None, raising=False)
    monkeypatch.setattr("semantic_search.index._INDEX_META", None, raising=False)


def test_get_index_rebuilds_when_tracked_file_mtime_changes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracked = tmp_path / "src" / "example.rs"
    tracked.parent.mkdir(parents=True)
    tracked.write_text("pub fn one() {}\n", encoding="utf-8")

    state = FakeIndexState()
    reset_index_globals(monkeypatch)
    install_fake_build(monkeypatch, state)
    monkeypatch.setattr("semantic_search.index.tracked_files", lambda _root: [tracked])
    monkeypatch.setattr("semantic_search.index.load_index", lambda _root: None)

    first = get_index(repo_root=tmp_path)
    assert first.reindexed is True
    assert state.build_count == 1

    tracked.write_text("pub fn two() {}\n", encoding="utf-8")

    second = get_index(repo_root=tmp_path)
    assert second.reindexed is True
    assert second.files_changed_since_index >= 1
    assert state.build_count == 2


def test_get_index_reuses_fresh_in_memory_index(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    tracked = tmp_path / "AGENTS.md"
    tracked.write_text("# Agents\n", encoding="utf-8")

    state = FakeIndexState()
    reset_index_globals(monkeypatch)
    install_fake_build(monkeypatch, state)
    monkeypatch.setattr("semantic_search.index.tracked_files", lambda _root: [tracked])
    monkeypatch.setattr("semantic_search.index.load_index", lambda _root: None)

    first = get_index(repo_root=tmp_path)
    second = get_index(repo_root=tmp_path)

    assert first.reindexed is True
    assert second.reindexed is False
    assert state.build_count == 1


def test_index_staleness_reports_changed_files(tmp_path: Path) -> None:
    tracked = tmp_path / "PROJECT.md"
    tracked.write_text("# Project\n", encoding="utf-8")
    mtimes = source_mtimes(tmp_path)
    meta = IndexMeta(
        built_at="2026-01-01T00:00:00+00:00",
        source_mtimes=mtimes,
        chunk_count=1,
    )
    assert index_staleness(tmp_path, meta)["files_changed_since_index"] == 0

    tracked.write_text("# Project updated\n", encoding="utf-8")
    staleness = index_staleness(tmp_path, meta)
    assert staleness["is_stale"] is True
    assert staleness["files_changed_since_index"] == 1


def test_cache_is_fresh_detects_added_file(tmp_path: Path) -> None:
    first = tmp_path / "AGENTS.md"
    first.write_text("# Agents\n", encoding="utf-8")
    cached = source_mtimes(tmp_path)

    second = tmp_path / "PROJECT.md"
    second.write_text("# Project\n", encoding="utf-8")

    assert cache_is_fresh(tmp_path, cached) is False
