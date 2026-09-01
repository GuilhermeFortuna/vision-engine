# VE-009 implementation plan: Tracker lifecycle and pipeline integration

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-009-tracker-lifecycle-and-integration-spec.md`](../specs/VE-009-tracker-lifecycle-and-integration-spec.md)  
**Depends on:** VE-008

## Current-system context

VE-006 through VE-008 supply the domain types, the filter, and the matcher. Each is
tested in isolation and none of them is wired into the frame loop. `main.rs` today
runs decode, infer, post-process, render. This pair inserts the tracker between
post-processing and rendering and gives identities their lifecycle.

## Interfaces produced

```rust
// src/tracking/mod.rs
pub struct Tracker { /* private: next_id: u64, entries: Vec<TrackEntry>, rejected_updates: u64 */ }

impl Tracker {
    pub fn new() -> Self;
    /// Runs one full frame of tracking and returns this frame's tracks.
    /// The returned Vec is owned by the caller and retained by nobody.
    pub fn update(&mut self, detections: &[Detection], stamp: FrameStamp) -> Vec<Track>;
    pub fn live_track_count(&self) -> usize;
    pub fn confirmed_track_count(&self) -> usize;
    pub fn rejected_updates(&self) -> u64;
}

// private
struct TrackEntry { track: Track, filter: KalmanBoxTracker }
```

## Implementation decisions

- `update` returns an owned `Vec<Track>` for the frame rather than a borrow into the
  tracker. The vector is dropped by the caller after rendering, which is what keeps
  per-frame state reclaimable; the tracker retains only `entries`.
- Fixed order inside `update`, with no early returns that would skip pruning:
  predict every entry and collect `(class_id, predicted_box)` in entry order;
  associate; apply matched updates; age unmatched entries; spawn tentative entries
  from unmatched detections; retire expired entries; build the return vector.
- Prediction is unconditional and happens exactly once per entry per frame, before
  association, and the association input is the predicted box rather than
  `track.bbox`. Keep the predicted boxes in a local vector indexed identically to
  `entries` so index translation from `Association` is direct.
- Matched entry: call `filter.update(detection_box)`. On `Applied`, set
  `track.bbox = filter.bbox()`. On `Rejected`, leave `track.bbox` as the predicted
  box and increment `self.rejected_updates`. In both cases the match still counts —
  set `confidence` from the detection, `last_seen = stamp`, `hits += 1`,
  `misses = 0`. A numerical rejection must not also cost the track its match.
- Promotion: after incrementing hits, a `Tentative` entry with
  `hits >= TRACK_PROMOTION_HITS` becomes `Confirmed`.
- Unmatched entry: `misses += 1` and `track.bbox = predicted_box`, so an occluded
  track keeps moving under its motion model rather than freezing.
- Retirement, evaluated after ageing, in this order:
  a `Tentative` entry with `misses > 0` is removed immediately; any entry with
  `stamp.media_ms - track.last_seen.media_ms > TRACK_RETENTION_MS` is removed.
  Removal uses `Vec::retain`, so the collection is compacted every frame.
  Retired ids are dropped and `next_id` only ever increases, so no id is reissued.
- Media-time retention needs one guard: when `stamp.media_ms` equals the previous
  frame's value because the clock adjusted a regression, the elapsed comparison
  still works because it is a difference against `last_seen`, not a per-frame
  decrement. No frame counter is needed.
- Spawn: an unmatched detection creates an entry with a fresh `TrackId`,
  `state: Tentative`, `first_seen = last_seen = stamp`, `hits = 1`, `misses = 0`,
  confidence from the detection, and a filter initialised from its box.
- The returned vector contains `Confirmed` and `Tentative` tracks. `Lost` entries are
  removed rather than returned. VE-010 decides how each state is drawn; VE-009 does
  not filter for the renderer.
- `main.rs` constructs one `Tracker` before the loop, alongside the detector, and
  times `tracker.update(...)` with `Instant::now()` around that call only, mirroring
  how decode and inference are already timed. The measurement is stored for VE-010
  to display.
- A tracker call cannot fail as a `Result`; all recoverable conditions are handled
  internally. `main.rs` therefore adds no new error path here beyond the frame index
  context already carried by the loop.

## Ordered implementation

1. Write a failing test building a `Tracker` and feeding a synthetic sequence of one
   detection moving linearly across ten frames, asserting the same `TrackId` on every
   frame after promotion and asserting the exact id value.
2. Implement enough of `update` — predict, associate, match, spawn — to pass it.
3. Write failing tests for promotion: a detection present for two frames is never
   `Confirmed`; present for three, it is `Confirmed` on exactly the third.
4. Write a failing test that a tentative track missing one frame is removed and that
   a later detection receives a new id, not the discarded one.
5. Implement promotion and tentative retirement.
6. Write a failing test for an occlusion gap shorter than `TRACK_RETENTION_MS` that
   preserves the id, and one for a gap longer than it that issues a new id. Drive
   media time explicitly through the `FrameStamp` values so the test does not depend
   on frame counts.
7. Implement ageing and media-time retirement.
8. Write a failing test where two same-class objects cross paths and assert both ids
   are preserved across the crossing, and a test where a rejected filter update
   leaves the track present, the frame successful, and `rejected_updates` at one.
9. Write a failing test for first-seen and last-seen: after a gap, `first_seen` is
   unchanged and `last_seen` is the most recent matched frame, both asserted by
   frame index.
10. Write a failing test that association receives predicted boxes: place a detection
    where the track will be predicted to move, not where it currently sits, and
    assert the match succeeds.
11. Write a failing test running five hundred synthetic frames of objects entering
    and leaving, asserting `live_track_count()` returns to zero during empty
    segments and never exceeds a stated bound.
12. Implement whatever remains until all pass.
13. Wire the tracker into `main.rs`: construct once, call per frame, time the call,
    keep the returned vector local to the iteration.
14. Run the full validation suite and confirm the release binary still plays a video
    end to end.

## Validation

- Unit: id stability, exact promotion frame, tentative discard, short and long gaps,
  same-class crossing, first-seen and last-seen, predicted-box association, rejected
  update tolerance, and bounded live track count over a long sequence.
- Integration: the release binary decodes, tracks, and exits cleanly on EOF and on
  Q, q, or Escape with tracking active.
- All tracker tests are pure and headless; none requires a video, model, or display.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx
```

## Handoff

Report the id sequences asserted by each deterministic scenario, the bound observed
for `live_track_count()` over the long sequence and whether it returned to zero,
the per-frame tracking latency range measured on the sample video, and any case
where a rejected filter update changed observable behavior.
