# VE-012 implementation plan: Pipeline stage extraction and serial baseline

**Status:** `READY` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-012-pipeline-stage-extraction-spec.md`](../specs/VE-012-pipeline-stage-extraction-spec.md)  
**Depends on:** VE-011

## Current-system context

`main.rs` is 801 lines carrying argument parsing, validation, capture setup, the
rolling frame-rate window, timestamp provenance counting, the playback loop, and the
end-of-run summary. `detector.rs` is 1153 lines carrying model loading, contract
validation, letterbox preprocessing, ONNX execution, candidate extraction, letterbox
inversion, and class-aware suppression. `render.rs` was already extracted in VE-010
and needs only to move and to gain window ownership.

The loop at `main.rs:352` is the thing being taken apart. Read it before starting:
every local it maintains across iterations (`media_offset_ms`, `fallback_reported`,
`last_stamp`, `last_progress_log`, `provenance_counts`) becomes state owned by one
stage, and getting that ownership wrong is how this extraction changes behavior.

`YoloV8Detector::infer` at `detector.rs:184` is preprocess, execute, and postprocess
in one function. It is split across two stages here. The important constraint is that
`extract_output_view` returns a view borrowing the `outputs` value, so postprocessing
must happen while that value is still alive. Returning an owned output array instead
would copy 84 x 8400 floats per frame, which is why the inference stage keeps
execution and postprocessing together.

## Interfaces produced

```rust
// src/cli.rs
pub struct Config {
    pub video: PathBuf,
    pub model: PathBuf,
    pub loop_for: Option<Duration>,
    pub track_dump: Option<PathBuf>,
    pub max_frames: Option<u64>,
}
pub enum ParseOutcome { Run(Config), Help }
pub fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<ParseOutcome>;
pub fn validate_config(config: &Config) -> Result<()>;
pub fn print_usage();

// src/detector.rs  (retains loading and contract validation only)
pub struct LoadedModel {
    pub session: Session,
    pub input_name: String,
    pub output_name: String,
}
impl LoadedModel { pub fn load(path: &Path) -> Result<Self>; }
pub struct Detection { pub class_id: u32, pub confidence: f32, pub bbox: BBox }
pub struct LetterboxTransform { /* unchanged */ }
pub fn coco_class_name(class_id: u32) -> Option<&'static str>;

// src/pipeline/decode.rs
pub struct DecodeStage { /* capture, clock, source_fps, loop_for, run_started,
                            media_offset_ms, provenance, fallback_reported */ }
pub enum DecodeOutcome { Frame { stamp: FrameStamp, decode_ms: f64 }, EndOfRun }
pub struct DecodeSummary {
    pub last_stamp: Option<FrameStamp>,
    pub frame_count: u64,
    pub reported: u64,
    pub derived_from_frame_rate: u64,
    pub derived_from_index: u64,
    pub adjustments: u64,
}
impl DecodeStage {
    pub fn open(video: &Path, loop_for: Option<Duration>) -> Result<Self>;
    pub fn next_into(&mut self, frame: &mut Mat) -> Result<DecodeOutcome>;
    pub fn summary(&self) -> DecodeSummary;
}

// src/pipeline/preprocess.rs
pub struct Prepared {
    pub input: Array4<f32>,
    pub transform: LetterboxTransform,
    pub preprocess_ms: f64,
}
pub fn prepare(frame: &Mat) -> Result<Prepared>;

// src/pipeline/infer.rs
pub struct InferStage { model: LoadedModel }
pub struct Detected { pub detections: Vec<Detection>, pub inference_ms: f64 }
impl InferStage {
    pub fn new(model: LoadedModel) -> Self;
    pub fn detect(&mut self, prepared: &Prepared) -> Result<Detected>;
}

// src/pipeline/track.rs
pub struct TrackStage { tracker: Tracker, last_progress_log: Option<Instant> }
pub struct Tracked { pub tracks: Vec<Track>, pub tracking_ms: f64 }
impl TrackStage {
    pub fn new() -> Self;
    pub fn update(&mut self, detections: &[Detection], stamp: FrameStamp)
        -> Result<Tracked>;
    pub fn live_track_count(&self) -> usize;
    pub fn rejected_updates(&self) -> u64;
}

// src/pipeline/render.rs   (moved from src/render.rs, plus window ownership)
pub struct RenderStage { /* window opened */ }
pub enum Presentation { Continue, QuitRequested }
impl RenderStage {
    pub fn open() -> Result<Self>;
    pub fn present(&mut self, frame: &mut Mat, tracks: &[Track], metrics: &FrameMetrics)
        -> Result<Presentation>;
    pub fn close(self) -> Result<()>;
}
pub fn draw_tracks(frame: &mut Mat, tracks: &[Track]) -> Result<()>;
pub fn draw_metrics_overlay(frame: &mut Mat, metrics: &FrameMetrics) -> Result<()>;

// src/pipeline/metrics.rs
pub struct FrameMetrics { /* unchanged fields from VE-010 */ }
pub struct RollingFps { /* moved from main.rs */ }

// src/pipeline/track_dump.rs
pub struct TrackDump { writer: BufWriter<File> }
impl TrackDump {
    pub fn create(path: &Path) -> Result<Self>;
    pub fn write_frame(&mut self, stamp: FrameStamp, tracks: &[Track]) -> Result<()>;
    pub fn finish(self) -> Result<()>;
}

// src/pipeline/mod.rs
pub fn run(config: &Config) -> Result<()>;
```

## Implementation decisions

- The extraction is done as a sequence of pure moves, each committed separately, with
  the full suite run between them. A reviewer must be able to read each commit and
  see that nothing changed but location. Combining moves into one commit destroys
  that property and is the main way this pair goes wrong.
- Stage functions take and return concrete values; the serial loop holds the frame,
  stamp, prepared tensor, detections, and tracks as locals and passes them along.
  Bundling these into message structs is deliberately deferred to VE-013, where the
  queue makes a single owned value per hop necessary. Introducing messages here would
  mean designing them for a caller that does not need them.
- `DecodeStage` owns everything the decoder needs to be self-contained: the capture,
  the `FrameClock`, the rewind offset, the provenance counts, and the run deadline.
  `next_into` writes into a caller-supplied `Mat`, preserving today's single-buffer
  reuse exactly. VE-014 changes this to a per-frame `Mat`; keeping the reuse here is
  what makes VE-012 a behavior-preserving move.
- `DecodeStage::next_into` returns `EndOfRun` for end of input, for the duration
  limit, and for the frame bound. The undecodable-first-frame case stays an error
  with its existing message. The rewind-and-continue path stays inside `next_into`,
  so looping is invisible to the caller, exactly as it is invisible to the loop body
  today.
- `LoadedModel` replaces `YoloV8Detector`. Its fields are public because the inference
  stage needs to run the session and then postprocess the borrowed output view in one
  scope. An accessor returning an owned array would add a 2.8 MB copy per frame.
- Preprocessing timing is newly separated out of the inference measurement. This is
  the one number that legitimately changes: `inference_ms` previously covered only
  `session.run` and still does, while `preprocess_ms` is newly reported and the
  overlay is unchanged in VE-012. Do not fold preprocessing into `inference_ms`.
- Progress logging moves to `TrackStage`, which owns the interval state. The
  end-of-run summary moves to `pipeline::run`, which assembles `DecodeSummary` and the
  tracker's rejected-update count. Field names, values, and message text in both logs
  stay byte-identical.
- Window ownership moves into `RenderStage`, whose `close` performs the destroy. The
  error-precedence rule from `main.rs:471` moves with it: a cleanup failure is
  reported only when the run itself succeeded, otherwise it is logged and the run's
  error stands.
- Track dump format is one line per track per frame, comma separated, with a header:
  `frame_index,media_ms,track_id,class_id,state,x_min,y_min,x_max,y_max,confidence`.
  Coordinates and media time use `{:.3}`, confidence `{:.4}`. Fixed precision is what
  makes a byte comparison meaningful; unformatted floats would make the file depend on
  float printing rather than on tracker behavior.
- All tracks are written, including `Lost`, with the state named. The renderer skips
  `Lost`, but the dump is a behavioral record rather than a screenshot, and a lifecycle
  regression that only affects lost tracks must still be visible.
- Tracks are written sorted by track identity ascending, so the file never depends on
  the tracker's internal storage order.
- The dump is written from the same slice the renderer is handed, at the same point in
  the loop, so it can never describe a different computation from the one displayed.
- `--max-frames` counts frames that completed tracking, and is checked in
  `DecodeStage` against the stamped count so it is enforced identically whether or not
  the run has looped.
- `src/render.rs` moves to `src/pipeline/render.rs`. Its drawing internals, constants,
  and tests are unchanged.

## Ordered implementation

1. Create the branch `VE-012-pipeline-stage-extraction-spec`.
2. Move `Config`, `ParseOutcome`, `parse_args`, `validate_config`,
   `validate_regular_file`, `print_usage`, and their tests verbatim into `src/cli.rs`.
   Wire `main.rs` to call them. Run the full suite. Commit the pure move.
3. Create `src/pipeline/mod.rs` and move `run_playback` into it as `run`, unchanged
   apart from its name and the imports it needs. Move `src/render.rs` to
   `src/pipeline/render.rs`. Run the full suite. Commit.
4. Move `RollingFps`, its tests, and `FrameMetrics` into `src/pipeline/metrics.rs`.
   Run the full suite. Commit.
5. Create `src/pipeline/decode.rs`. Move `open_video_capture`, `read_capture_fps`,
   `read_capture_pos_msec`, `sanitize_capture_f64`, `frame_interval_ms`,
   `video_path_for_opencv`, `classify_playback_frame`, `ProvenanceCounts`, and their
   tests into it. Build `DecodeStage` around them, moving the loop's decode block,
   rewind path, timestamp stamping, fallback warning, and duration check into
   `next_into`. Leave `pipeline::run` calling `next_into` in place of that block.
6. Run the full suite and the release binary on the sample video. Confirm the logs,
   including the fallback warning and the summary line, are identical to the previous
   commit. Commit.
7. Write a failing test asserting `DecodeStage::open` on a directory and on a missing
   file produce the existing error messages, and that an undecodable file produces the
   existing `video file could not be decoded` error. Make them pass without changing
   the messages. Commit.
8. Create `src/pipeline/preprocess.rs`. Move `preprocess_frame` and every
   preprocessing test out of `detector.rs`, wrap it as `prepare` with its own timing,
   and have `pipeline::run` call it. Commit.
9. Create `src/pipeline/infer.rs`. Move `extract_output_view`, `output_shape_matches`,
   `extract_candidates`, `best_class_score`, `xywh_to_corners`,
   `inverse_letterbox_coordinate`, `clamp_coordinate`, `restore_to_source`,
   `postprocess_output`, the suppression helpers, `sort_deterministic`, and their
   tests out of `detector.rs`. Build `InferStage::detect` as: run the session, time
   only that call, extract the view, postprocess in the same scope. Commit.
10. Rename `YoloV8Detector` to `LoadedModel` in `detector.rs`, reducing it to loading,
    contract validation, class names, `Detection`, and `LetterboxTransform`. Run the
    full suite. Commit.
11. Create `src/pipeline/track.rs` with `TrackStage`, moving the tracker call, the
    tracking timing, and the progress-logging interval out of the loop. Commit.
12. Move window creation and destruction into `RenderStage::open` and `close`, and the
    draw, display, and key-poll sequence into `present`. Move `should_exit` with it.
    Preserve the draw-tracks-then-metrics order and the cleanup error precedence.
    Run the release binary and confirm quitting with `q` and with Escape still returns
    success. Commit.
13. Confirm `pipeline::run` is now a loop of six calls plus the summary, and that
    `main.rs` contains only `main`, `run`, `init_tracing`, and the top-level error
    report. Commit any remaining tidy-up.
14. Write a failing test for the track-dump line format: a confirmed track at frame 7
    renders exactly
    `7,233.333,42,0,confirmed,10.000,20.000,110.000,220.000,0.9100`, and a frame with
    two tracks emits them in ascending identity order regardless of the order they are
    passed in.
15. Run it and confirm it fails.
16. Implement `TrackDump` with the header line, fixed formatting, and the sort. Run
    the test and confirm it passes. Commit.
17. Add `--track-dump <path>` and `--max-frames <n>` to `src/cli.rs` with failing
    tests first: both parse, both default to absent, a missing value is an error, a
    zero frame count is an error, and an unparseable frame count is an error. Mirror
    the existing `--loop-for-seconds` error wording. Commit.
18. Wire `--track-dump` into `pipeline::run`, writing each frame's tracks after the
    renderer receives them, and `--max-frames` into `DecodeStage`. Commit.
19. Write a failing integration test that runs the binary twice over the same input
    with `--track-dump` and asserts the two files are byte-identical, and a test that
    `--max-frames 5` produces exactly five distinct frame indices in the dump.
20. Make them pass. Commit.
21. Run the release binary on the sample video with and without `--track-dump` and
    confirm the rendering, logs, and exit code are unchanged with the option absent.
22. Produce the baseline. Create `docs/development/baselines/VE-012/` containing:
    `single-pass.csv` from one complete pass over the sample video;
    `looped.csv` from a run with `--loop-for-seconds` large enough not to bind and
    `--max-frames` set to roughly one and a half times the single-pass frame count, so
    the run crosses a loop boundary; and `baseline.md`.
23. Record throughput: five sustained runs via `scripts/sustained-run.sh`, reporting
    every run's frames per second and the median. Add to `baseline.md` the sample
    video path and size, the model path, both exact command lines, the frame counts,
    the commit hash, the CPU model, and the core count.
24. Verify the baseline reproduces: regenerate both dumps and confirm they are
    byte-identical to the committed files. Commit the baseline.
25. Run the full validation suite.

## Validation

- Unit: existing detector, tracking, rendering, and argument tests pass unmoved in
  behavior; new tests for `DecodeStage` error messages, track-dump formatting and
  ordering, and the two new arguments.
- Integration: two runs over one input produce byte-identical dumps; `--max-frames`
  bounds the run exactly; existing `tests/cli_exit.rs` and
  `tests/tracking_acceptance.rs` pass unchanged.
- Manual: the release binary on the sample video renders as before, logs the same
  lines, and exits with the same codes for end of input, `q`, and Escape.
- Regression: each extraction commit is behavior-identical to its predecessor.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --track-dump /tmp/a.csv
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --track-dump /tmp/b.csv
diff /tmp/a.csv /tmp/b.csv
scripts/sustained-run.sh samples/test.mp4 models/yolov8n.onnx
```

## Handoff

Report the line counts of `main.rs` and `detector.rs` before and after, the final
module layout, confirmation that each extraction commit is a pure move, the two
baseline dumps' frame counts and sizes, the five throughput values and their median,
and the environment recorded in `baseline.md`. Note explicitly that `preprocess_ms` is
newly separated from `inference_ms` and that no other reported value changed.
