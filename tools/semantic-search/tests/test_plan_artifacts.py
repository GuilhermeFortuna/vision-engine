from __future__ import annotations

from pathlib import Path

from semantic_search.extract_plan_artifacts import (
    build_plan_artifacts,
    extract_paths,
    extract_symbols,
    parse_plan_artifacts,
)


def test_extract_paths_normalizes_relative_and_script_names() -> None:
    text = """
`pipeline/render.rs` and `sustained-run.sh` plus `src/pipeline/metrics.rs`.
"""
    paths = extract_paths(text)
    assert "src/pipeline/render.rs" in paths
    assert "src/pipeline/metrics.rs" in paths
    assert "scripts/sustained-run.sh" in paths


def test_extract_symbols_includes_structs_and_snake_case() -> None:
    text = """
`RunStats` and `queue_depths` and `METRICS_AREA_BOTTOM`.
pub struct FrameMetrics {}
pub fn percentile(sorted_samples: &[f64], fraction: f64) -> f64;
"""
    symbols = extract_symbols(text)
    assert "RunStats" in symbols
    assert "queue_depths" in symbols
    assert "METRICS_AREA_BOTTOM" in symbols
    assert "FrameMetrics" in symbols
    assert "percentile" in symbols


def test_parse_plan_artifacts_from_markdown(tmp_path: Path) -> None:
    rel_path = "docs/development/plans/VE-015-pipeline-instrumentation-plan.md"
    file_path = tmp_path / rel_path
    file_path.parent.mkdir(parents=True)
    file_path.write_text(
        """# VE-015 plan

## Interfaces produced

```rust
// src/pipeline/metrics.rs
pub struct RunStats {}
pub fn percentile(sorted_samples: &[f64], fraction: f64) -> f64;
```

`sustained-run.sh` references `METRICS_AREA_BOTTOM`.
""",
        encoding="utf-8",
    )
    parsed = parse_plan_artifacts(tmp_path, rel_path)
    assert parsed is not None
    assert parsed.ve_id == "VE-015"
    assert "src/pipeline/metrics.rs" in parsed.files
    assert "scripts/sustained-run.sh" in parsed.files
    assert "RunStats" in parsed.symbols
    assert "METRICS_AREA_BOTTOM" in parsed.symbols


def test_build_plan_artifacts_merges_spec_and_plan(tmp_path: Path) -> None:
    spec = tmp_path / "docs/development/specs/VE-013-frame-messages-spec.md"
    plan = tmp_path / "docs/development/plans/VE-013-frame-messages-plan.md"
    spec.parent.mkdir(parents=True)
    plan.parent.mkdir(parents=True)
    spec.write_text("# VE-013 spec\n\n`src/pipeline/message.rs`\n", encoding="utf-8")
    plan.write_text(
        "# VE-013 plan\n\n`src/pipeline/queue.rs` and `DecodedFrame`\n",
        encoding="utf-8",
    )
    artifacts = build_plan_artifacts(tmp_path)
    item = artifacts["VE-013"]
    assert item.spec_path is not None
    assert item.plan_path.endswith("VE-013-frame-messages-plan.md")
    assert "src/pipeline/message.rs" in item.files
    assert "src/pipeline/queue.rs" in item.files
    assert "DecodedFrame" in item.symbols
