"""tree-sitter based Rust chunk extraction."""

from __future__ import annotations

import re
from pathlib import Path

import tree_sitter_rust as tsrust
from tree_sitter import Language, Node, Parser

from semantic_search.chunks import Chunk

RUST_LANGUAGE = Language(tsrust.language())
PARSER = Parser(RUST_LANGUAGE)

CFG_TEST_MOD = re.compile(r"^#\[cfg\(test\)\]\s*\n\s*mod tests\b", re.MULTILINE)
CFG_TEST_IMPL = re.compile(r"^#\[cfg\(test\)\]\s*\n\s*impl\b", re.MULTILINE)


def module_path_for_file(rel_path: str) -> str:
    path = Path(rel_path)
    if path.suffix == ".rs":
        path = path.with_suffix("")
    parts = list(path.parts)
    if parts and parts[0] == "src":
        parts = parts[1:]
    if parts and parts[-1] == "mod":
        parts = parts[:-1]
    elif parts and parts[-1] == "lib":
        parts = parts[:-1]
    return "::".join(parts)


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


def line_number(source: str, byte_offset: int) -> int:
    return source.count("\n", 0, byte_offset) + 1


def node_text(source: str, node: Node) -> str:
    return source[node.start_byte : node.end_byte]


def leading_doc_comment(source: str, node: Node) -> str:
    start_byte = node.start_byte
    prefix = source[:start_byte]
    lines: list[str] = []
    for line in reversed(prefix.splitlines()):
        stripped = line.strip()
        if stripped.startswith("///") or stripped.startswith("//!"):
            lines.append(stripped.lstrip("/").strip())
            continue
        if not stripped:
            continue
        break
    lines.reverse()
    return "\n".join(lines)


def has_cfg_test_attribute(source: str, node: Node) -> bool:
    start_byte = node.start_byte
    prefix = source[:start_byte].rstrip()
    if not prefix:
        return False
    last_line = prefix.rsplit("\n", 1)[-1].strip()
    return last_line == "#[cfg(test)]"


def function_name(node: Node, source: str) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    return node_text(source, name_node)


def type_name(node: Node, source: str) -> str | None:
    name_node = node.child_by_field_name("name")
    if name_node is None:
        return None
    if name_node.type == "type_identifier":
        return node_text(source, name_node)
    if name_node.type == "identifier":
        return node_text(source, name_node)
    return node_text(source, name_node)


def impl_target(node: Node, source: str) -> str:
    type_node = node.child_by_field_name("type")
    if type_node is None:
        return "impl"
    return node_text(source, type_node)


def collect_impl_signatures(node: Node, source: str) -> str:
    signatures: list[str] = []
    body = node.child_by_field_name("body")
    if body is None:
        return ""
    for child in body.children:
        if child.type != "function_item":
            continue
        signature = node_text(source, child).split("{", 1)[0].strip()
        if signature:
            signatures.append(signature)
    return "\n".join(signatures)


def extract_chunks_from_source(rel_path: str, source: str) -> list[Chunk]:
    module_path = module_path_for_file(rel_path)
    production_source = strip_test_sections(source)
    tree = PARSER.parse(production_source.encode("utf-8"))
    chunks: list[Chunk] = []

    def walk(node: Node) -> None:
        if node.type == "function_item":
            if not has_cfg_test_attribute(production_source, node):
                name = function_name(node, production_source)
                if name:
                    chunks.append(
                        Chunk(
                            path=rel_path,
                            symbol=name,
                            kind="fn",
                            line_start=line_number(production_source, node.start_byte),
                            line_end=line_number(production_source, node.end_byte),
                            signature=node_text(production_source, node)
                            .split("{", 1)[0]
                            .strip(),
                            doc=leading_doc_comment(production_source, node),
                            body=node_text(production_source, node),
                            module_path=module_path,
                        )
                    )
        elif node.type == "struct_item":
            name = type_name(node, production_source)
            if name:
                chunks.append(
                    Chunk(
                        path=rel_path,
                        symbol=name,
                        kind="struct",
                        line_start=line_number(production_source, node.start_byte),
                        line_end=line_number(production_source, node.end_byte),
                        signature=node_text(production_source, node)
                        .split("{", 1)[0]
                        .strip(),
                        doc=leading_doc_comment(production_source, node),
                        body=node_text(production_source, node),
                        module_path=module_path,
                    )
                )
        elif node.type == "enum_item":
            name = type_name(node, production_source)
            if name:
                chunks.append(
                    Chunk(
                        path=rel_path,
                        symbol=name,
                        kind="enum",
                        line_start=line_number(production_source, node.start_byte),
                        line_end=line_number(production_source, node.end_byte),
                        signature=node_text(production_source, node)
                        .split("{", 1)[0]
                        .strip(),
                        doc=leading_doc_comment(production_source, node),
                        body=node_text(production_source, node),
                        module_path=module_path,
                    )
                )
        elif node.type == "impl_item":
            if not has_cfg_test_attribute(production_source, node):
                target = impl_target(node, production_source)
                signatures = collect_impl_signatures(node, production_source)
                chunks.append(
                    Chunk(
                        path=rel_path,
                        symbol=target,
                        kind="impl",
                        line_start=line_number(production_source, node.start_byte),
                        line_end=line_number(production_source, node.end_byte),
                        signature=f"impl {target}",
                        doc=leading_doc_comment(production_source, node),
                        body=signatures,
                        module_path=module_path,
                    )
                )

        for child in node.children:
            walk(child)

    walk(tree.root_node)

    if not chunks:
        chunks.append(
            Chunk(
                path=rel_path,
                symbol=Path(rel_path).stem,
                kind="mod",
                line_start=1,
                line_end=max(1, production_source.count("\n") + 1),
                signature=f"mod {module_path}",
                doc="",
                body=production_source[:500],
                module_path=module_path,
            )
        )

    symbols_by_file: dict[str, list[str]] = {}
    for chunk in chunks:
        symbols_by_file.setdefault(chunk.path, []).append(chunk.symbol)
    for chunk in chunks:
        chunk.neighbors = sorted(
            symbol for symbol in symbols_by_file.get(chunk.path, []) if symbol != chunk.symbol
        )

    return chunks


def extract_rust_chunks(repo_root: Path) -> list[Chunk]:
    chunks: list[Chunk] = []
    for rust_file in sorted(repo_root.glob("src/**/*.rs")):
        rel_path = rust_file.relative_to(repo_root).as_posix()
        source = rust_file.read_text(encoding="utf-8")
        chunks.extend(extract_chunks_from_source(rel_path, source))
    return chunks
