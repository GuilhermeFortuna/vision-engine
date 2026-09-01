"""Extract VE task touchpoints from specs and plans."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from semantic_search.chunks import Chunk

REPO_ROOT = Path(__file__).resolve().parents[4]
CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
PLAN_ARTIFACTS_FILE = CACHE_DIR / "plan_artifacts.json"

VE_ID_RE = re.compile(r"VE-\d{3}", re.IGNORECASE)
PATH_RE = re.compile(r"(?:src|scripts)/[\w./_-]+\.(?:rs|sh)")
RELATIVE_PATH_RE = re.compile(r"(?:pipeline|src/pipeline)/[\w./_-]+\.rs")
BARE_SCRIPT_RE = re.compile(r"(?<![\w./-])([\w-]+\.sh)(?![\w./-])")
BACKTICK_SYMBOL_RE = re.compile(r"`([A-Za-z_][\w]*)`")
RUST_SYMBOL_RE = re.compile(
    r"(?:pub\s+)?(?:struct|fn|impl|const|enum)\s+(\w+)",
    re.MULTILINE,
)
TYPE_METHOD_RE = re.compile(r"\b([A-Z][A-Za-z0-9_]*)(::[a-z_][\w]*)")
RUST_BLOCK_RE = re.compile(r"```rust\n(.*?)```", re.DOTALL)

GENERIC_SYMBOLS = frozenset(
    {
        "new",
        "open",
        "default",
        "run",
        "main",
        "mod",
        "lib",
        "test",
        "tests",
        "impl",
        "struct",
        "fn",
        "const",
        "enum",
        "self",
        "mut",
        "pub",
        "crate",
        "use",
        "let",
        "return",
        "true",
        "false",
        "None",
        "Some",
        "Ok",
        "Err",
        "Result",
        "Option",
        "Vec",
        "String",
        "usize",
        "u64",
        "f64",
        "i32",
        "bool",
        "BLOCKED",
        "OpenCV",
    }
)


@dataclass
class PlanArtifacts:
    ve_id: str
    files: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)
    plan_path: str = ""
    spec_path: str | None = None


def ve_id_from_path(path: str) -> str | None:
    match = VE_ID_RE.search(path)
    if not match:
        return None
    return match.group(0).upper()


def normalize_path(path: str) -> str:
    cleaned = path.strip().strip("`").strip()
    if cleaned.startswith("src/") or cleaned.startswith("scripts/"):
        return cleaned
    if cleaned.startswith("pipeline/"):
        return f"src/{cleaned}"
    if cleaned.endswith(".sh"):
        return f"scripts/{cleaned}"
    return cleaned


def extract_paths(text: str) -> list[str]:
    paths: list[str] = []
    for match in PATH_RE.findall(text):
        paths.append(normalize_path(match))
    for match in RELATIVE_PATH_RE.findall(text):
        paths.append(normalize_path(match))
    for match in BARE_SCRIPT_RE.findall(text):
        paths.append(normalize_path(match))
    return dedupe(path for path in paths if path)


def dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def is_useful_symbol(symbol: str) -> bool:
    if symbol in GENERIC_SYMBOLS:
        return False
    if len(symbol) > 40:
        return False
    if symbol.count("_") >= 4 and symbol.islower():
        return False
    return True


def extract_backtick_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for match in BACKTICK_SYMBOL_RE.findall(text):
        if match in GENERIC_SYMBOLS:
            continue
        if match[0].isupper() or "_" in match:
            symbols.append(match)
    return symbols


def extract_rust_symbols(text: str) -> list[str]:
    symbols: list[str] = []
    for match in RUST_SYMBOL_RE.findall(text):
        if match not in GENERIC_SYMBOLS:
            symbols.append(match)
    for match in TYPE_METHOD_RE.findall(text):
        type_name = match[0]
        if type_name not in GENERIC_SYMBOLS:
            symbols.append(type_name)
    return symbols


def extract_symbols(text: str) -> list[str]:
    symbols = extract_backtick_symbols(text)
    symbols.extend(extract_rust_symbols(text))
    return symbols


def extract_plan_symbols(source: str) -> list[str]:
    symbols = extract_backtick_symbols(source)
    rust_blocks = "\n".join(RUST_BLOCK_RE.findall(source))
    symbols.extend(extract_rust_symbols(rust_blocks))
    for line in source.splitlines():
        if "::" in line:
            symbols.extend(extract_rust_symbols(line))
    return symbols


def enrich_artifacts(item: PlanArtifacts) -> PlanArtifacts:
    files = list(item.files)
    symbols = [symbol for symbol in item.symbols if is_useful_symbol(symbol)]
    if "Shutdown" in symbols and "src/pipeline/queue.rs" not in files:
        files.append("src/pipeline/queue.rs")
    if any(symbol in symbols for symbol in ("send", "recv", "bounded", "Shutdown")):
        if "src/pipeline/queue.rs" not in files:
            files.append("src/pipeline/queue.rs")
    return PlanArtifacts(
        ve_id=item.ve_id,
        files=dedupe(files),
        symbols=dedupe(symbols),
        plan_path=item.plan_path,
        spec_path=item.spec_path,
    )


def parse_plan_artifacts(repo_root: Path, rel_path: str) -> PlanArtifacts | None:
    ve_id = ve_id_from_path(rel_path)
    if ve_id is None:
        return None

    source = (repo_root / rel_path).read_text(encoding="utf-8")
    files = extract_paths(source)
    symbols = extract_plan_symbols(source)

    if "/specs/" in rel_path:
        return PlanArtifacts(
            ve_id=ve_id,
            files=files,
            symbols=symbols,
            spec_path=rel_path,
        )
    return PlanArtifacts(
        ve_id=ve_id,
        files=files,
        symbols=symbols,
        plan_path=rel_path,
    )


def merge_artifacts(left: PlanArtifacts, right: PlanArtifacts) -> PlanArtifacts:
    return PlanArtifacts(
        ve_id=left.ve_id,
        files=dedupe(left.files + right.files),
        symbols=dedupe(left.symbols + right.symbols),
        plan_path=left.plan_path or right.plan_path,
        spec_path=left.spec_path or right.spec_path,
    )


def build_plan_artifacts(repo_root: Path = REPO_ROOT) -> dict[str, PlanArtifacts]:
    artifacts: dict[str, PlanArtifacts] = {}
    patterns = [
        "docs/development/specs/VE-*.md",
        "docs/development/plans/VE-*.md",
    ]
    for pattern in patterns:
        for path in sorted(repo_root.glob(pattern)):
            rel_path = path.relative_to(repo_root).as_posix()
            parsed = parse_plan_artifacts(repo_root, rel_path)
            if parsed is None:
                continue
            existing = artifacts.get(parsed.ve_id)
            if existing is None:
                artifacts[parsed.ve_id] = enrich_artifacts(parsed)
            else:
                artifacts[parsed.ve_id] = enrich_artifacts(merge_artifacts(existing, parsed))
    return {ve_id: enrich_artifacts(item) for ve_id, item in artifacts.items()}


def artifacts_to_dict(artifacts: dict[str, PlanArtifacts]) -> dict[str, object]:
    return {
        ve_id: {
            "ve_id": item.ve_id,
            "files": item.files,
            "symbols": item.symbols,
            "plan_path": item.plan_path,
            "spec_path": item.spec_path,
        }
        for ve_id, item in sorted(artifacts.items())
    }


def artifacts_from_dict(payload: dict[str, object]) -> dict[str, PlanArtifacts]:
    artifacts: dict[str, PlanArtifacts] = {}
    for ve_id, raw in payload.items():
        if not isinstance(raw, dict):
            continue
        artifacts[str(ve_id)] = PlanArtifacts(
            ve_id=str(raw.get("ve_id", ve_id)),
            files=[str(path) for path in raw.get("files", [])],
            symbols=[str(symbol) for symbol in raw.get("symbols", [])],
            plan_path=str(raw.get("plan_path", "")),
            spec_path=str(raw["spec_path"]) if raw.get("spec_path") else None,
        )
    return artifacts


def save_plan_artifacts(artifacts: dict[str, PlanArtifacts]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_ARTIFACTS_FILE.write_text(
        json.dumps(artifacts_to_dict(artifacts), indent=2),
        encoding="utf-8",
    )


def load_plan_artifacts() -> dict[str, PlanArtifacts]:
    if not PLAN_ARTIFACTS_FILE.exists():
        return {}
    payload = json.loads(PLAN_ARTIFACTS_FILE.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return artifacts_from_dict(payload)


def plan_ref_chunks(artifacts: dict[str, PlanArtifacts]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for item in artifacts.values():
        for path in item.files:
            chunks.append(
                Chunk(
                    path=path,
                    symbol=Path(path).stem,
                    kind="plan_ref",
                    line_start=1,
                    line_end=1,
                    signature=f"plan_ref {item.ve_id} file {path}",
                    doc=f"Named in {item.ve_id} plan/spec",
                    body=f"task: {item.ve_id}\nfile: {path}",
                    module_path="",
                    ve_id=item.ve_id,
                )
            )
        for symbol in item.symbols:
            file_hint = next(
                (path for path in item.files if symbol.lower() in path.lower()),
                item.files[0] if item.files else "",
            )
            chunks.append(
                Chunk(
                    path=file_hint or f"docs/development/plans/{item.ve_id.lower()}.md",
                    symbol=symbol,
                    kind="plan_ref",
                    line_start=1,
                    line_end=1,
                    signature=f"plan_ref {item.ve_id} symbol {symbol}",
                    doc=f"Named in {item.ve_id} interfaces produced",
                    body=f"task: {item.ve_id}\nsymbol: {symbol}",
                    module_path="",
                    ve_id=item.ve_id,
                )
            )
    return chunks
