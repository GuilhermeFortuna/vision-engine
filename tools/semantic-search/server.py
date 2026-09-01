"""Semantic code search MCP server for Vision Engine Rust sources."""

from __future__ import annotations

import json
import math
import re
import urllib.request
from pathlib import Path
from typing import NamedTuple

from mcp.server.mcpserver import MCPServer

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_GLOB = "src/**/*.rs"
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
EMBED_MODEL = "qwen3-embedding:4b"

FN_START = re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?fn\s+(\w+)\b", re.MULTILINE)
CFG_TEST_MOD = re.compile(r"^#\[cfg\(test\)\]\s*\n\s*mod tests\b", re.MULTILINE)
CFG_TEST_IMPL = re.compile(r"^#\[cfg\(test\)\]\s*\n\s*impl\b", re.MULTILINE)
CFG_TEST_ATTR = re.compile(r"#\[cfg\(test\)\]\s*$")


class FunctionChunk(NamedTuple):
    path: str
    name: str
    code: str
    embedding: list[float]


def embed_texts(texts: list[str]) -> list[list[float]]:
    payload = json.dumps({"model": EMBED_MODEL, "input": texts}).encode()
    request = urllib.request.Request(
        OLLAMA_EMBED_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        body = json.loads(response.read())
    return body["embeddings"]


def embed_text(text: str) -> list[float]:
    return embed_texts([text])[0]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def strip_test_sections(source: str) -> str:
    match = CFG_TEST_MOD.search(source)
    if match:
        source = source[: match.start()]

    while True:
        match = CFG_TEST_IMPL.search(source)
        if not match:
            break
        brace_start = source.find("{", match.end())
        if brace_start == -1:
            source = source[: match.start()]
            break
        depth = 0
        end = brace_start
        for index in range(brace_start, len(source)):
            char = source[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        source = source[: match.start()] + source[end:]

    return source


def has_cfg_test_attribute(source: str, fn_start: int) -> bool:
    prefix = source[:fn_start].rstrip()
    if not prefix:
        return False
    last_line = prefix.rsplit("\n", 1)[-1].strip()
    return bool(CFG_TEST_ATTR.search(last_line))


def extract_functions(source: str) -> list[tuple[str, str]]:
    production_source = strip_test_sections(source)
    functions: list[tuple[str, str]] = []

    for match in FN_START.finditer(production_source):
        if has_cfg_test_attribute(production_source, match.start()):
            continue

        brace_start = production_source.find("{", match.end())
        if brace_start == -1:
            continue

        depth = 0
        end = brace_start
        for index in range(brace_start, len(production_source)):
            char = production_source[index]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break
        else:
            continue

        functions.append((match.group(1), production_source[match.start() : end]))

    return functions


def build_index() -> list[FunctionChunk]:
    chunks: list[FunctionChunk] = []
    rust_files = sorted(REPO_ROOT.glob(SRC_GLOB))

    pending: list[tuple[str, str, str]] = []
    for rust_file in rust_files:
        rel_path = rust_file.relative_to(REPO_ROOT).as_posix()
        source = rust_file.read_text(encoding="utf-8")
        for name, code in extract_functions(source):
            pending.append((rel_path, name, code))

    if not pending:
        return chunks

    texts = [f"{path}\n{name}\n{code}" for path, name, code in pending]
    embeddings = embed_texts(texts)

    for (path, name, code), embedding in zip(pending, embeddings, strict=True):
        chunks.append(
            FunctionChunk(path=path, name=name, code=code, embedding=embedding)
        )

    return chunks


INDEX = build_index()

mcp = MCPServer("semantic-search")


@mcp.tool()
def semantic_code_search(query: str, top_k: int = 3) -> list[dict[str, object]]:
    """Search production Rust functions by semantic similarity."""
    if top_k < 1:
        return []

    query_embedding = embed_text(query)
    ranked = sorted(
        INDEX,
        key=lambda chunk: cosine_similarity(query_embedding, chunk.embedding),
        reverse=True,
    )[:top_k]

    return [
        {
            "path": chunk.path,
            "function": chunk.name,
            "score": round(cosine_similarity(query_embedding, chunk.embedding), 4),
            "code": chunk.code,
        }
        for chunk in ranked
    ]


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
