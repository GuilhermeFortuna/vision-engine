"""Chunk model and text builders for indexing and search."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Chunk:
    path: str
    symbol: str
    kind: str
    line_start: int
    line_end: int
    signature: str = ""
    doc: str = ""
    body: str = ""
    module_path: str = ""
    neighbors: list[str] = field(default_factory=list)
    ve_id: str | None = None
    embedding: list[float] | None = None

    def search_text(self) -> str:
        parts = [
            self.path,
            self.module_path,
            self.kind,
            self.symbol,
            self.signature,
            self.doc,
            self.body,
            " ".join(self.neighbors),
        ]
        if self.ve_id:
            parts.append(self.ve_id)
        return "\n".join(part for part in parts if part)

    def embedding_text(self) -> str:
        lines = [
            f"module: {self.module_path}",
            f"kind: {self.kind}",
            f"symbol: {self.symbol}",
        ]
        if self.signature:
            lines.append(f"signature: {self.signature}")
        if self.doc:
            lines.append(f"doc: {self.doc}")
        if self.neighbors:
            lines.append(f"neighbors: {', '.join(self.neighbors)}")
        if self.ve_id:
            lines.append(f"task: {self.ve_id}")
        if self.body:
            lines.append(f"body: {self.body}")
        return "\n".join(lines)

    def display_code(self) -> str:
        if self.body:
            return self.body
        if self.signature:
            return self.signature
        return self.doc

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "symbol": self.symbol,
            "kind": self.kind,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "signature": self.signature,
            "doc": self.doc,
            "body": self.body,
            "module_path": self.module_path,
            "neighbors": self.neighbors,
            "ve_id": self.ve_id,
            "embedding": self.embedding,
        }

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Chunk:
        neighbors = data.get("neighbors", [])
        return cls(
            path=str(data["path"]),
            symbol=str(data["symbol"]),
            kind=str(data["kind"]),
            line_start=int(data["line_start"]),
            line_end=int(data["line_end"]),
            signature=str(data.get("signature", "")),
            doc=str(data.get("doc", "")),
            body=str(data.get("body", "")),
            module_path=str(data.get("module_path", "")),
            neighbors=list(neighbors) if isinstance(neighbors, list) else [],
            ve_id=str(data["ve_id"]) if data.get("ve_id") else None,
            embedding=list(data["embedding"]) if data.get("embedding") else None,
        )
