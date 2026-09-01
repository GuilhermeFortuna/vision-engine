from __future__ import annotations

from pathlib import Path

from semantic_search.extract_docs import chunk_markdown_file, ve_id_from_path
from semantic_search.extract_rust import extract_chunks_from_source, module_path_for_file


def test_module_path_for_pipeline_file() -> None:
    assert module_path_for_file("src/pipeline/queue.rs") == "pipeline::queue"


def test_extract_rust_finds_struct_and_function() -> None:
    source = """
/// Frame with stamp.
pub struct DecodedFrame {
    pub stamp: u64,
}

pub fn prepare(frame: DecodedFrame) -> Result<()> {
    Ok(())
}
"""
    chunks = extract_chunks_from_source("src/pipeline/message.rs", source)
    symbols = {chunk.symbol: chunk.kind for chunk in chunks}
    assert symbols["DecodedFrame"] == "struct"
    assert symbols["prepare"] == "fn"


def test_extract_docs_finds_ve_id_and_sections(tmp_path: Path) -> None:
    rel_path = "docs/development/specs/VE-013-frame-messages-and-bounded-queues-spec.md"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True)
    file_path.write_text(
        "# VE-013\n\n## Purpose\n\nBuild frame messages.\n\n## Requirements\n\nQueues block.\n",
        encoding="utf-8",
    )
    chunks = chunk_markdown_file(tmp_path, rel_path)
    assert ve_id_from_path(rel_path) == "VE-013"
    assert any(chunk.symbol == "purpose" for chunk in chunks)
    assert all(chunk.ve_id == "VE-013" for chunk in chunks)
