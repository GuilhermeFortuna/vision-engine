# VE-016 implementation plan: Milestone 3 hardening and acceptance

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-016-milestone-three-hardening-and-acceptance-spec.md`](../specs/VE-016-milestone-three-hardening-and-acceptance-spec.md)  
**Depends on:** VE-015

## Current-system context

The pipeline runs on four threads with a main-thread renderer, reports per-stage
latency and queue depth, and has matched the serial baseline once, in VE-014, over a
single pass. What has not been established is that it holds: that parity survives a
looped run and repeated runs, that it is faster by a measured margin rather than an
impression, and that it cannot deadlock or grow without bound.

Concurrency defects do not appear on demand. Every check here is built to provoke a
failure rather than to wait for one, which is why the liveness work stalls stages
deliberately and exercises shutdown from each state that could hang it.

## Interfaces produced

```rust
// tests/pipeline_liveness.rs
/// Blocks a named stage for a set number of frames so backpressure can be observed.
fn stall_stage(stage: &'static str, frames: u64, duration: Duration) -> StallHandle;

// src/pipeline/runtime.rs  (test-only, extending VE-014's fault injection)
#[cfg(test)]
impl Pipeline {
    pub(crate) fn stall_stage_for(&self, stage: &'static str, delay: Duration);
    pub(crate) fn peak_queue_depths(&self) -> QueueDepths;
}
```

No production interface changes are expected. Any that this pair needs is a finding to
report, not an assumed step.

## Implementation decisions

- Parity is re-verified here rather than inherited from VE-014, over both committed
  baselines and across repeated runs. VE-014 proved the pipeline could match once; this
  proves it does so reliably, which is a different claim and the one the milestone
  rests on.
- Repeated-run parity is checked by running the threaded pipeline five times and
  comparing the dumps to each other as well as to the baseline. Two threaded runs
  differing from each other is the clearest possible signal of a race, and it would be
  invisible to a single comparison against the baseline.
- If parity fails, the first differing line's frame index localizes it and the fix
  belongs in the stage that owns that frame's state. The baseline is not regenerated
  and no assertion is loosened. A parity failure here means the refactor changed
  behavior, which is precisely what this batch was structured to detect.
- Throughput is measured on the same machine, input, and model as VE-012, with both
  commits recorded, because a comparison across machines or inputs measures nothing.
  Five runs per configuration with individual values and medians reported; a single
  pair of numbers cannot distinguish an improvement from run-to-run variance.
- The expected result is a throughput ratio bounded by the slowest stage: with stages
  overlapping, the pipeline's rate approaches the slowest stage's rate rather than the
  sum of all five. If inference dominates, the realistic gain is the decode,
  preprocess, track, and render time that now overlaps it. Stating this in advance is
  what keeps the measured result honest — a number far above it means the measurement
  is wrong, not that the pipeline is exceptional.
- Per-frame allocation cost is isolated by timing frame allocation directly over a
  representative run and reporting it as a share of frame time. It is the known way
  this refactor could lose throughput and it must be quantified even if the overall
  result is positive, because it is the evidence a later buffer-pool task would argue
  from.
- Stall testing injects a delay into one stage at a time. The assertions are that the
  queues stay within capacity, that resident memory does not grow with the stall's
  length, and that a shutdown request during the stall still terminates and joins every
  thread. A pipeline that stalls safely but cannot be stopped is still broken.
- Shutdown is exercised from five states — full queues, empty queues, mid-frame, at end
  of input, and before the first frame. These are the states where a missed wakeup
  hides. Each asserts termination and that all four threads are joined, and none may
  rely on a timeout to pass.
- Start and stop cycling runs the pipeline ten times in one process, checking thread
  count and resident memory afterwards, which catches a stage thread that outlives its
  pipeline.
- Optimization is out of scope unless throughput fails to improve. If that happens, the
  bottleneck named by the instrumentation drives one targeted change, measured before
  and after, and anything further is recorded as a follow-up rather than pursued.

## Ordered implementation

1. Create the branch `VE-016-milestone-three-hardening-and-acceptance-spec`.
2. Write the failing parity test over both baselines: a single-pass run matches
   `docs/development/baselines/VE-012/single-pass.csv`, and a frame-bounded run using
   the baseline's `--max-frames` value matches
   `docs/development/baselines/VE-012/looped.csv`, both byte for byte.
3. Run it. If the looped case fails, check first whether the media-time offset applied
   at rewind is computed from the last stamp the decoder produced rather than from a
   stamp that has already moved downstream — that state crossed a stage boundary in
   VE-013 and is the most likely divergence. Fix and commit.
4. Write the failing repeated-run test: five threaded runs produce five byte-identical
   dumps. Make it pass. Commit.
5. Write the failing stall tests, one per stage: with a stage stalled for two seconds,
   every queue stays within capacity, resident memory does not grow with the stall
   length, and a shutdown request during the stall terminates the run and joins every
   thread.
6. Run them and fix what they expose. Commit.
7. Write the failing shutdown-state tests: from full queues, from empty queues, from
   mid-frame, at end of input, and before the first frame. Each asserts termination and
   four joined threads, with no timeout as the passing mechanism. Make them pass.
   Commit.
8. Write the failing start-and-stop test: ten pipeline lifecycles in one process leave
   no additional threads and no monotonic memory growth. Make it pass. Commit.
9. Re-verify failure behavior across every stage: induced failure and induced panic
   each return non-zero and name the stage; a frame with no detections, a frame with no
   live tracks, a rejected filter update, and a source with no usable timestamps are
   all non-fatal. Commit any fixes.
10. Verify exit codes: help, end of input, `q`, and Escape return success; a missing
    file, a directory as input, an undecodable file, and an unsupported model return
    non-zero without a panic or a backtrace.
11. Measure throughput: five sustained runs of the pipeline, recording each value and
    the median, alongside VE-012's committed serial figures. Record both commits, the
    CPU model, and the core count.
12. Measure per-frame allocation cost over a representative run and express it as a
    share of mean frame time.
13. Run the sustained soak: `scripts/sustained-run.sh` for its full duration, recording
    the resident-memory series, the queue-depth series, and the per-stage latency
    series. Confirm no unbounded growth and no depth beyond capacity at any sample.
14. If throughput did not improve, name the bottleneck from the instrumentation, make
    one targeted change, measure before and after, and record any further candidates as
    follow-ups without pursuing them.
15. Re-measure identity churn on the pipeline using VE-011's method, and compare with
    the figure VE-011 recorded.
16. Run the release binary on the sample video and confirm visually that boxes track
    their objects, identities persist, and the full overlay is readable.
17. Write `docs/development/baselines/VE-016/acceptance.md` with every measurement: the
    parity results, the throughput values and ratio, the per-stage breakdown, the queue
    saturation pattern and named bottleneck, the allocation cost, the memory and depth
    series, the churn comparison, and the environment.
18. Update `docs/development/STATUS.md` to mark VE-012 through VE-016 `DONE` and record
    Milestone 3 as accepted, referencing the acceptance record.
19. Record the follow-ups this batch deliberately deferred — buffer pooling, inference
    worker pools, and frame-drop policy for live sources — with the measurements that
    would justify each.
20. Run the full validation suite.

## Validation

- Parity: both baselines matched byte for byte; five threaded runs identical to each
  other.
- Liveness: per-stage stall tests; five shutdown-state tests; ten start-and-stop
  cycles; full-duration soak with no unbounded growth and no queue beyond capacity.
- Failure: induced failure and induced panic per stage; the four non-fatal operating
  conditions; the full exit-code matrix.
- Measurement: five throughput runs per configuration with individual values and
  medians; per-stage latency breakdown; queue saturation; per-frame allocation cost;
  memory series; identity churn against VE-011.
- Manual visual: tracking correctness and overlay legibility on the sample video.
- No liveness test may pass by timing out. A test that needs a timeout has found a
  defect.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release
cargo build --release
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --track-dump /tmp/ve016-single.csv
diff /tmp/ve016-single.csv docs/development/baselines/VE-012/single-pass.csv
scripts/sustained-run.sh samples/test.mp4 models/yolov8n.onnx
```

## Handoff

Report the parity outcome for both baselines and across repeated runs, the throughput
values and ratio with both commits and the machine stated, the per-stage breakdown with
the bottleneck named, the per-frame allocation cost, the soak's memory and queue-depth
series, the identity churn against VE-011's figure, every liveness case exercised and
its result, and the deferred follow-ups with their justifying measurements. State
plainly whether Milestone 3 is accepted, and if any acceptance criterion did not pass,
say which and what it measured rather than describing the milestone as complete.
