# VE-014 implementation plan: Threaded pipeline runtime

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-014-threaded-pipeline-runtime-spec.md`](../specs/VE-014-threaded-pipeline-runtime-spec.md)  
**Depends on:** VE-013

## Current-system context

After VE-013 the stages consume and produce owned messages, and a tested bounded queue
with a shutdown signal exists but is unused by the binary. `pipeline::run` still calls
the five stages in sequence. This pair replaces that sequence with four threads and a
main-thread renderer.

Everything that makes this safe is already in place: each stage owns its resources, the
messages are `Send`, and the queue drains before reporting disconnection. What remains
is spawning, wiring, and the two things concurrency adds that a serial loop never had —
termination from either end, and errors arriving from a thread rather than a call.

## Interfaces produced

```rust
// src/pipeline/runtime.rs
pub struct StageHandle {
    name: &'static str,
    join: JoinHandle<Result<()>>,
}

pub struct Pipeline {
    stages: Vec<StageHandle>,
    tracked_rx: Receiver<TrackedFrame>,
    shutdown: Shutdown,
}

impl Pipeline {
    /// Spawns decode, preprocess, infer, and track. The renderer stays with the
    /// caller, which must be the main thread.
    pub fn spawn(config: &Config, model: LoadedModel) -> Result<Self>;

    /// Blocks for the next tracked frame. `Ok(None)` means the pipeline drained.
    pub fn next_tracked(&self) -> Option<TrackedFrame>;

    pub fn request_shutdown(&self);

    /// Joins every stage. Returns the first failure in stage order, logging any
    /// others. A panicked stage becomes an error naming that stage.
    pub fn join(self) -> Result<()>;
}

// src/pipeline/mod.rs
pub fn run(config: &Config) -> Result<()>;
```

## Implementation decisions

- Four spawned threads, named `decode`, `preprocess`, `infer`, and `track`, each
  running the same shape: receive, do the work, send, repeat until either end
  disconnects. The renderer runs on the main thread because OpenCV's window and key
  polling require it, which is a constraint on the design rather than a choice.
- Each stage moves its resources into its thread at spawn: `DecodeStage` owns the
  capture and clock, `InferStage` owns the model session, `TrackStage` owns the
  tracker. Nothing in the frame path is shared or locked, so there is no lock ordering
  to reason about and no contention to profile.
- Frame ordering needs no mechanism. Each stage is a single thread reading and writing
  first-in-first-out queues, so decode order is preserved structurally. This is the
  reason the spec forbids worker pools in M3: a pool would need a reorder buffer, and
  the parity requirement would then rest on that buffer being correct.
- Shutdown has exactly two directions and both must be implemented.
  Downstream: the decoder returns `None`, drops its sender, each stage drains its queue
  and then observes disconnection, finishes, and drops its own sender. Every in-flight
  frame is processed. Upstream: `Shutdown::request` from the renderer or from a failing
  stage wakes every blocked queue end, which then reports disconnection.
- The decoder also checks `Shutdown::is_requested` each iteration, because it is the
  one stage that can be busy producing rather than blocked on a queue.
- Stage bodies return `Result<()>`. On error a stage requests shutdown before returning,
  so the failure propagates in both directions rather than waiting for the queues to
  drain naturally.
- `join` collects every stage's outcome in fixed stage order — decode, preprocess,
  infer, track — and reports the first error found, logging the rest at error level.
  Fixed stage order is used rather than wall-clock order because a shutdown cascade
  makes downstream failures near-simultaneous with the cause, and the upstream stage is
  the one that explains them.
- A panicked thread is caught by `join` returning `Err`, and is converted into an error
  naming the stage and reporting the panic payload when it is a string. It must never
  become a hang: the panicking thread's queue ends are dropped as its stack unwinds,
  which disconnects its neighbours.
- Error context keeps VE-011's discipline: every stage error is wrapped with the stage
  name and the frame index being processed.
- The renderer's loop calls `next_tracked` and stops when it returns `None`, on
  `QuitRequested`, or on a render error. It then requests shutdown, drops its receiver,
  and joins. Requesting shutdown on the normal path too is harmless and removes a case
  from the reasoning: joining always follows a requested shutdown.
- Window cleanup and error precedence are unchanged from VE-012: `RenderStage::close`
  runs regardless, and its failure is reported only if nothing else failed.
- Fault injection for the failure tests is compiled in under `#[cfg(test)]` on
  `Pipeline::spawn` — a per-stage "fail at frame N" and "panic at frame N" switch.
  Testing failure propagation by corrupting real inputs would exercise the decoder's
  error paths only, leaving three stages unverified.

## Ordered implementation

1. Create the branch `VE-014-threaded-pipeline-runtime-spec`.
2. Write the failing end-to-end ordering test first, against a `Pipeline` that does not
   exist yet: spawn the pipeline over a synthetic source of 50 frames, collect every
   tracked frame from `next_tracked`, and assert the stamps' indices are exactly 0
   through 49 in order with none missing or repeated.
3. Run it and confirm it fails to compile.
4. Create `src/pipeline/runtime.rs` with `StageHandle` and `Pipeline::spawn`, wiring
   four queues of `QUEUE_CAPACITY` and spawning the four stage threads. Implement
   `next_tracked` as a `recv` returning `None` on disconnection, `request_shutdown`,
   and a first-cut `join` that collects results in stage order.
5. Rewrite `pipeline::run` to spawn the pipeline, drive the renderer on the main
   thread, and join. Run the ordering test until it passes. Commit.
6. Run the release binary on the sample video. Confirm it renders, that `q` and Escape
   quit, and that end of input exits with success. Commit any fixes.
7. Write the failing drain test: over a 50-frame synthetic source, the number of frames
   the renderer receives equals the number the decoder produced, with no in-flight
   frame lost at end of input. Make it pass. Commit.
8. Write the failing parity test:

```rust
#[test]
fn threaded_run_matches_the_serial_baseline() {
    // Runs the release-equivalent pipeline over the baseline input with a track dump
    // and compares against docs/development/baselines/VE-012/single-pass.csv.
    let dump = run_pipeline_to_dump(BASELINE_VIDEO, BASELINE_MODEL);
    let baseline = std::fs::read_to_string(
        "docs/development/baselines/VE-012/single-pass.csv").unwrap();
    assert_eq!(dump, baseline, "threaded output diverged from the serial baseline");
}
```

9. Run it. If it fails, treat the difference as the defect: find the frame index of the
   first differing line and work back from it. Do not regenerate the baseline. The
   likely causes, in order to check: the tracker receiving frames out of order, the
   media-time offset being recomputed in the wrong stage after a loop, or a frame
   dropped at shutdown. Commit when it passes.
10. Add the `#[cfg(test)]` fault-injection switches to `Pipeline::spawn`.
11. Write failing tests, one per stage, that an induced failure at frame 10 makes the
    run return an error naming that stage and including frame index 10, and that every
    thread is joined. Make them pass. Commit.
12. Write failing tests, one per stage, that an induced panic at frame 10 returns an
    error naming that stage and does not hang. Make them pass. Commit.
13. Write a failing test that a shutdown requested while the queues are full — induced
    by a renderer that stops consuming — terminates the run and joins every thread.
    Make it pass. Commit.
14. Write a failing test that a shutdown requested before the first frame terminates
    cleanly. Make it pass. Commit.
15. Confirm concurrency is real rather than nominal: run the release binary and check
    that the observed frames per second exceeds the serial baseline's, and that the sum
    of the per-stage times per frame now exceeds the wall-clock time per frame. Record
    both numbers; they are the evidence that work overlaps.
16. Verify the exit codes are unchanged: help returns success, end of input returns
    success, `q` and Escape return success, a missing file returns non-zero, an
    undecodable file returns non-zero.
17. Run the full validation suite.

## Validation

- Unit and integration: frame ordering across 50 frames; frames rendered equals frames
  decoded; induced failure in each of the four stages plus the renderer; induced panic
  in each stage; shutdown with full queues; shutdown before the first frame; every
  thread joined in every case.
- Parity: the threaded track dump is byte-identical to
  `docs/development/baselines/VE-012/single-pass.csv`.
- Regression: `tests/cli_exit.rs` and `tests/tracking_acceptance.rs` pass unchanged.
- Every test must terminate on its own logic. A test that needs a timeout to finish is
  reporting a defect in the runtime, not a slow machine.
- Run the concurrency tests in release as well as debug; thread interleavings that
  debug builds hide often appear only under optimization.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
cargo build --release
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --track-dump /tmp/ve014.csv
diff /tmp/ve014.csv docs/development/baselines/VE-012/single-pass.csv
```

## Handoff

Report the parity result explicitly as a byte comparison against the committed
baseline, the observed frames per second against the serial baseline figure, the
evidence that stages overlap, which fault-injection cases were exercised and their
reported errors, confirmation that every termination path joins all four threads, and
any place where the shutdown cascade needed a mechanism beyond queue disconnection and
the shutdown flag.
