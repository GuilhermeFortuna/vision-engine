# VE-011 implementation plan: Milestone 2 hardening and acceptance

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-011-milestone-two-hardening-and-acceptance-spec.md`](../specs/VE-011-milestone-two-hardening-and-acceptance-spec.md)  
**Depends on:** VE-010

## Current-system context

VE-006 through VE-010 deliver a working tracker with visible identities. Each pair
tested its own unit. This pair tests the assembled tracker as one thing, closes the
failure paths tracking introduced, and produces the recorded evidence Milestone 2 is
accepted on. Milestone 1's VE-005 acceptance run is the baseline these numbers are
compared against.

## Interfaces produced

```rust
// tests/tracking_acceptance.rs — deterministic, no video/model/display
fn run_sequence(frames: &[Vec<Detection>], fps: f64) -> Vec<Vec<Track>>;
fn ids_of(tracks: &[Track]) -> Vec<u64>;         // confirmed only, ascending
fn moving_box(frame: usize, class_id: u32) -> Detection;
```

```bash
# scripts/sustained-run.sh — wraps the release binary for the long run
# Loops the input, samples RSS/frames/tracks once per minute, writes CSV.
```

## Implementation decisions

- The deterministic scenarios live in one integration test file driving `Tracker`
  directly through synthetic detection frames. They construct `FrameStamp` values
  explicitly at a stated frame rate so occlusion gaps are expressed in media time and
  do not depend on how fast the test machine runs.
- The five gating scenarios assert identities by value with `assert_eq!` on a
  `Vec<u64>`. A scenario that only counts tracks does not satisfy the spec and must
  not be written. They are: continuous tracking, a gap shorter than retention
  preserving the id, a gap longer than retention issuing a new id, two
  different-class objects overlapping, and a single-frame spurious detection that
  never confirms.
- The same-class crossing is a **diagnostic**, not a gate. Build it as two boxes
  converging, overlapping for two frames, then separating. Assert only that the run
  completes without error and that two confirmed tracks exist both before and after
  the crossing. Then compute whether the id pair survived — compare the two ids
  observed before the crossing against the two after — and print the result through
  `--nocapture` for the handoff. Do not `assert!` on identity preservation in either
  direction: vanilla SORT has no appearance or re-identification signal, so an
  ambiguous same-class crossing is genuinely undecidable from geometry alone, and a
  gating assertion here would either fail correct work or get weakened until it
  passed.
- Record the diagnostic outcome as a number in the handoff, not as a verdict. If ids
  switch, that is the measurement that would justify appearance features or
  second-stage association in a later milestone.
- Failure-path work is an audit, not a redesign: walk every new call site from
  VE-006 through VE-010 and confirm each carries context naming the stage and frame
  index. The conditions the spec calls normal — rejected filter updates, zero
  detections, zero live tracks, absent media timestamps — get an explicit test each
  asserting the run continues and no error is returned.
- The sustained run is a shell wrapper rather than Rust code, because it observes the
  process from outside. It loops the sample video to reach twelve minutes, sleeps in
  sixty-second intervals, and on each wake samples `VmRSS` from `/proc/<pid>/status`
  plus the frame and track counters the binary logs, appending one CSV row. Sampling
  starts at the two-minute mark and ends at the twelve-minute mark inclusive, so a
  complete run emits exactly eleven rows. The script exits non-zero if it collects
  fewer, so a short or crashed run cannot be mistaken for a passing one. No benchmark
  framework is added.
- To expose track counts to the sampler, the binary logs live and confirmed track
  counts alongside the frame count at a fixed interval. This is a small, permanent
  observability addition, not test-only scaffolding.
- Pass criteria are evaluated from the CSV and copied verbatim into the handoff:
  final RSS within five per cent or ten megabytes of the first post-warm-up sample,
  whichever is larger; the final five samples not strictly increasing; live track
  count bounded and falling toward zero on empty segments. A rising live track count
  fails the run regardless of RSS.
- Identity churn for the acceptance segment is computed as distinct confirmed ids
  issued during the segment divided by frames in the segment, times one hundred. The
  expected object count is stated by the operator beforehand, not inferred afterward,
  so the comparison cannot be rationalised.

## Ordered implementation

1. Create `tests/tracking_acceptance.rs` with the helpers above and the continuous
   tracking scenario, asserting exact ids across twenty frames.
2. Add the short-gap and long-gap scenarios, driving media time explicitly.
3. Add the different-class overlap scenario as a gating assertion, and the
   same-class crossing as the diagnostic described above — weak invariants asserted,
   identity outcome printed rather than asserted.
4. Add the spurious single-frame detection scenario, asserting no confirmed id ever
   appears.
5. Run all six. The five gating scenarios must pass; investigate any failure among
   them as a tracker finding before changing a test. Record the diagnostic's printed
   outcome for the handoff whichever way it goes.
6. Audit error context across every VE-006 to VE-010 call site; add missing context
   naming stage and frame index.
7. Add tests for the four normal-condition cases, each asserting the run continues
   without error.
8. Add the periodic frame and track-count log line to the binary.
9. Write `scripts/sustained-run.sh`: loop input, sample once per minute, emit CSV
   with elapsed, frames, RSS, live tracks, confirmed tracks.
10. Run the full validation suite and the release build.
11. Execute the manual failure matrix: EOF, Q, q, Escape, missing video, unreadable
    video, missing model, invalid ONNX, unsupported tensor shape, and a source with
    no usable media timestamps. Record status and message for each.
12. Execute the acceptance run: state the expected object count for the designated
    segment beforehand, then record observed confirmed ids and churn.
13. Execute the sustained run: twelve minutes total, sampled every sixty seconds
    from the two-minute mark to the twelve-minute mark, yielding eleven rows.
    Evaluate the ten measured samples against the pass criteria.
14. Compare decode, inference, and FPS against VE-005's Milestone 1 baseline and
    report tracking's added per-frame cost.
15. Review the complete batch 02 diff for Milestone 3 scope, then update `STATUS.md`
    only if every criterion has passing evidence.

## Validation

- Automated: five gating identity scenarios asserting exact ids, one recorded
  same-class crossing diagnostic, four
  normal-condition tests, the full existing unit and integration suite, and startup
  failure coverage — all headless.
- Manual: the ten-case failure matrix, and visual confirmation of persistent
  identities with all five metrics displayed.
- Measured: acceptance-segment identity churn against a pre-stated expected count,
  and the sustained-run CSV evaluated against the stated RSS and track-count criteria.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
./scripts/sustained-run.sh samples/test.mp4 models/yolov8n.onnx --minutes 12 --interval 60
```

## Handoff

Report every validation command and manual scenario as pass, fail, or blocked, never
as assumed. Include release hardware, video resolution and duration, model identity,
decode, inference and tracking latency ranges, processing FPS, the expected and
observed identity counts with the resulting churn, and all eleven sustained-run samples
with the pass criteria evaluated against the ten measured ones, and the same-class
crossing diagnostic outcome. State tracking's added per-frame cost
against the Milestone 1 baseline. Mark VE-011 and Milestone 2 complete only when all
criteria have evidence; do not begin Milestone 3.
