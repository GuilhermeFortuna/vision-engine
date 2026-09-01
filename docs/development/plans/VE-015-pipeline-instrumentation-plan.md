# VE-015 implementation plan: Pipeline instrumentation

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-015-pipeline-instrumentation-spec.md`](../specs/VE-015-pipeline-instrumentation-spec.md)  
**Depends on:** VE-014

## Current-system context

The pipeline runs concurrently and each message already carries `StageTimings` filled
in by the stage that produced it, so per-frame latency is measured but never reported.
The overlay still shows VE-010's five values, sized by `METRICS_AREA_BOTTOM` at 160
pixels in `pipeline/render.rs`, and `sustained-run.sh` still records the five columns
VE-011 defined. `Receiver::len` and `Receiver::capacity` exist from VE-013 and are
unused.

The pipeline's frame rate is now set by its slowest stage, and nothing reports which
one that is. That is the gap this pair closes.

## Interfaces produced

```rust
// src/pipeline/metrics.rs
pub struct FrameMetrics {
    pub timings: StageTimings,     // decode, preprocess, inference, tracking
    pub render_ms: f64,
    pub queue_depths: QueueDepths,
    pub fps: Option<f64>,
    pub confirmed_tracks: usize,
}

pub struct QueueDepths {
    pub decoded: (usize, usize),   // (depth, capacity), one per queue
    pub prepared: (usize, usize),
    pub detected: (usize, usize),
    pub tracked: (usize, usize),
}

/// Accumulates per-stage latency and per-queue depth across a run for the summary.
pub struct RunStats { /* per-stage sample vectors, per-queue depth counters */ }
impl RunStats {
    pub fn new() -> Self;
    pub fn record(&mut self, metrics: &FrameMetrics);
    pub fn summary(&self) -> RunSummary;
}

pub struct StageStat { pub name: &'static str, pub mean_ms: f64, pub p95_ms: f64 }
pub struct QueueStat {
    pub name: &'static str,
    pub mean_depth: f64,
    pub fraction_at_capacity: f64,
}
pub struct RunSummary {
    pub stages: [StageStat; 5],
    pub queues: [QueueStat; 4],
    pub slowest_stage: &'static str,
    pub frames: u64,
}

pub fn percentile(sorted_samples: &[f64], fraction: f64) -> f64;
pub fn format_ms(value: f64) -> String;
pub fn format_depth(depth: usize, capacity: usize) -> String;

// src/pipeline/runtime.rs
impl Pipeline { pub fn queue_depths(&self) -> QueueDepths; }
```

## Implementation decisions

- Stage latencies ride on the message, which VE-013 already arranged. With four frames
  in flight at once, a renderer that timed the stages itself would report four
  different frames' work as one frame's, so the only correct place for these numbers is
  the message.
- `render_ms` is the one timing the renderer measures itself, covering drawing,
  display, and the key poll for the previous frame. It is attached at the renderer
  rather than the message because no later stage consumes it.
- Queue depths are sampled by the renderer once per frame through
  `Pipeline::queue_depths`. Depth is a property of the pipeline at an instant, not of a
  frame, and presenting a sampled value as a per-frame measurement would misrepresent
  it. The summary reports how many samples sat at capacity rather than an average
  alone, because saturation is the signal and a mean of 1.4 out of 2 hides it.
- Depth renders as `depth/capacity`, for example `2/2`, so a saturated queue is
  readable without knowing the capacity. Saturation immediately upstream of a stage
  identifies that stage as the bottleneck; a queue sitting at `0/2` says its producer
  is the constraint.
- The summary names the slowest stage by highest mean latency. Naming it is not a
  convenience: it is the number VE-016's acceptance and every later optimization
  argues from, and leaving it to be inferred from five figures invites it to be
  inferred wrongly.
- Percentile uses nearest-rank on sorted samples, stated explicitly so the reported p95
  means one thing across runs. Samples are retained per stage for the run; at typical
  run lengths this is a few hundred thousand floats, well within budget, and it avoids
  an approximate estimator whose error would have to be characterized.
- The overlay gains a stage block and a queue block. `METRICS_AREA_BOTTOM` rises from
  160 to 280 at the established 30-pixel line spacing, and `label_origin` from VE-010
  keeps track labels clear of it without further change.
- The frames-per-second line is labelled as throughput at the renderer. Under
  backpressure the decoder is deliberately idle, so a reader who takes this number for
  a decode rate will conclude the decoder is slow when it is being held back on
  purpose.
- The end-of-run summary keeps every field it emits today — frames, media time, the
  three provenance counts, adjustments, and rejected updates — and adds the stage and
  queue figures. Existing fields are not renamed or reordered, so a reader comparing
  against a VE-011 log still can.
- `sustained-run.sh` gains its new columns at the end of the existing header, for the
  same reason.
- Instrumentation cost is measured, not assumed. `Instant::now` around five stages plus
  four atomic depth reads per frame should be far below a millisecond, but the claim is
  checked against VE-014's throughput and reported either way.

## Ordered implementation

1. Create the branch `VE-015-pipeline-instrumentation-spec`.
2. Write failing unit tests for `percentile`: the p95 of 1 through 100 is 95; a
   single-sample slice returns that sample; an empty slice returns 0.0.
3. Run them and confirm they fail. Implement `percentile` with nearest-rank. Confirm
   they pass. Commit.
4. Write failing unit tests for `format_depth` and `format_ms`: a full capacity-2 queue
   renders `2/2`, an empty one `0/2`; `format_ms` renders one decimal place and renders
   an unavailable value as `--` rather than as a zero.
5. Implement both. Confirm the tests pass. Commit.
6. Add `render_ms` and `queue_depths` to `FrameMetrics`, and `Pipeline::queue_depths`
   reading `len` and `capacity` from the four receivers. Commit.
7. Write a failing test for `RunStats::summary`: feed three frames with known stage
   timings where preprocessing is slowest, and assert `slowest_stage` is
   `"preprocess"`, the means match, and a queue observed at capacity in two of three
   samples reports `fraction_at_capacity` of 2/3.
8. Implement `RunStats` and `summary`. Confirm the test passes. Commit.
9. Extend the overlay with the five stage lines and the four queue lines, raise
   `METRICS_AREA_BOTTOM` to 280, and label the frame rate as renderer throughput. Run
   the release binary and confirm all values are readable simultaneously and that
   track labels at every frame edge stay clear of the overlay. Commit.
10. Extend the end-of-run summary log with the per-stage means and p95s, the per-queue
    mean depth and fraction at capacity, and the named slowest stage, keeping every
    existing field unchanged. Commit.
11. Extend `scripts/sustained-run.sh` to parse and record the new values, appending
    columns after the existing five. Confirm a full sustained run produces a
    well-formed record with the new columns populated. Commit.
12. Demonstrate the bottleneck signal on a real run: capture a run where the inference
    queue sits at capacity and the others do not, and confirm the summary names the
    stage the depths imply. Record the output.
13. Measure instrumentation cost: five sustained runs with the instrumentation in
    place, compared against VE-014's recorded figures, reporting individual values and
    medians for both. If throughput drops measurably, reduce what is sampled per frame
    rather than leaving the cost unexplained.
14. Confirm the track dump still matches the VE-012 baseline byte for byte —
    instrumentation must not change behavior.
15. Run the full validation suite.

## Validation

- Unit: `percentile` at p95, single sample, and empty; `format_ms` including the
  unavailable case; `format_depth` at empty and at capacity; `RunStats::summary` for
  means, p95, saturation fraction, and slowest-stage selection.
- Manual visual: all fourteen values readable at once; labels clear of the overlay at
  every frame edge and corner; a saturated queue distinguishable from a starved one at
  a glance.
- Integration: a sustained run produces the new columns; the summary log retains every
  VE-011 field.
- Regression: the track dump matches the VE-012 baseline; existing tests pass
  unchanged.
- Measurement: throughput with instrumentation against VE-014, five runs each,
  individual values and medians.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx
scripts/sustained-run.sh samples/test.mp4 models/yolov8n.onnx
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --track-dump /tmp/ve015.csv
diff /tmp/ve015.csv docs/development/baselines/VE-012/single-pass.csv
```

## Handoff

Report the per-stage latency breakdown and the named slowest stage from a real run, the
queue saturation pattern observed and what it implies about the bottleneck, the
throughput comparison against VE-014 with individual values and medians, the
instrumentation cost as a percentage, confirmation that the overlay remains legible and
labels stay clear of it, and confirmation that the track dump is unchanged.
