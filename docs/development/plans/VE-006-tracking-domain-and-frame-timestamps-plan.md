# VE-006 implementation plan: Tracking domain model and frame timestamps

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-006-tracking-domain-and-frame-timestamps-spec.md`](../specs/VE-006-tracking-domain-and-frame-timestamps-spec.md)  
**Depends on:** VE-005

## Current-system context

`src/detector.rs` defines `Detection` with four loose `f32` corner fields and a
free function `intersection_over_union(&Detection, &Detection)`. `src/main.rs`
owns the frame loop and currently derives no time value at all. There is no
`src/tracking/` module. This pair adds the module, moves box geometry into a
shared type, and gives every frame a stamp.

## Interfaces produced

```rust
// src/tracking/mod.rs
pub mod clock;
pub mod params;
pub mod track;

// src/tracking/track.rs
pub struct BBox { pub x_min: f32, pub y_min: f32, pub x_max: f32, pub y_max: f32 }
impl BBox {
    pub fn from_center_size(cx: f32, cy: f32, w: f32, h: f32) -> Self;
    pub fn width(&self) -> f32;
    pub fn height(&self) -> f32;
    pub fn area(&self) -> f32;
    pub fn center(&self) -> (f32, f32);
    pub fn aspect_ratio(&self) -> f32;          // width / height
    pub fn is_valid(&self) -> bool;             // finite, positive extent
    pub fn clamped_to(&self, w: f32, h: f32) -> Option<Self>;
    pub fn iou(&self, other: &Self) -> f32;
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TrackId(pub u64);                    // Display writes `#42`

pub enum TrackState { Tentative, Confirmed, Lost }

pub struct Track {
    pub id: TrackId,
    pub class_id: u32,
    pub state: TrackState,
    pub bbox: BBox,
    pub confidence: f32,
    pub first_seen: FrameStamp,
    pub last_seen: FrameStamp,
    pub hits: u32,
    pub misses: u32,
}

// src/tracking/clock.rs
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeSource { Reported, DerivedFromFrameRate, DerivedFromIndex }

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FrameStamp {
    pub index: u64,
    pub media_ms: f64,
    pub source: TimeSource,
    pub adjusted: bool,
}

pub struct FrameClock { /* private */ }
impl FrameClock {
    pub fn new(source_fps: Option<f64>) -> Self;
    pub fn stamp(&mut self, reported_ms: Option<f64>) -> FrameStamp;
    pub fn adjustments(&self) -> u64;
}

// src/detector.rs (modified)
pub struct Detection { pub class_id: u32, pub confidence: f32, pub bbox: BBox }
```

## Implementation decisions

- `FrameClock::stamp` takes the reported media time as an `Option<f64>` supplied by
  the caller. The clock itself never touches OpenCV, which keeps it fully testable
  and keeps the backend-specific read in `main.rs` where the capture lives. This is
  what satisfies the spec's requirement that no particular property-read ordering
  becomes the contract.
- Resolution order inside `stamp`: use the reported value when it is `Some`, finite
  and non-negative (`TimeSource::Reported`); otherwise `index / source_fps * 1000`
  when the source frame rate is `Some`, finite and greater than zero
  (`DerivedFromFrameRate`); otherwise `index * (1000.0 / NOMINAL_FPS)`
  (`DerivedFromIndex`).
- Monotonicity is enforced last: if the resolved value is less than the previous
  frame's value, clamp it to the previous value, set `adjusted: true`, and increment
  the adjustment counter. `main.rs` logs the counter once at end of run.
- `main.rs` reads the reported time with `capture.get(videoio::CAP_PROP_POS_MSEC)`
  and the frame rate with `CAP_PROP_FPS`, mapping non-finite, negative, or zero
  results to `None`. Which side of `read()` the position is sampled on is decided by
  observing the actual backend during step 6 and recorded in the handoff — it is not
  fixed by the spec.
- `BBox` replaces the four corner fields on `Detection` and absorbs
  `intersection_over_union` as the `iou` method. `postprocess_output`,
  `restore_to_source`, `non_maximum_suppression`, `sort_deterministic`, and
  `draw_detections` are updated to the new field access. The numeric behavior of
  each is unchanged, so their existing assertions continue to hold.
- Tracker parameters live in `src/tracking/params.rs` as three constants, each with
  a comment giving its justification:
  `TRACK_PROMOTION_HITS = 3` (a detection must survive three frames before its
  identity is presented as stable), `TRACK_RETENTION_MS = 1000.0` (roughly one
  second of occlusion tolerance, expressed in media time so behavior does not change
  with source frame rate or with the unpaced loop), and
  `ASSOCIATION_IOU_GATE = 0.30` (conventional SORT gate; retained pending VE-011
  evidence).
- `Track` is a plain data carrier in this pair. It gains no methods that mutate
  lifecycle state; VE-009 owns transitions.

## Ordered implementation

1. Create `src/tracking/` with `mod.rs`, `track.rs`, `clock.rs`, and `params.rs`,
   and declare `mod tracking;` in `main.rs`.
2. Write failing unit tests for `BBox` geometry: `from_center_size` round-trips,
   `area`, `aspect_ratio`, `is_valid` rejecting non-finite and zero-extent boxes,
   `clamped_to` returning `None` when a box clamps to empty, and `iou` reproducing
   the identical, disjoint, and partial-overlap values already asserted in
   `detector.rs`.
3. Implement `BBox` until those tests pass.
4. Move `Detection` onto `BBox`, delete the free `intersection_over_union`, and
   update every call site in `detector.rs` and `main.rs`. Run the existing detector
   test suite and confirm the same assertions still pass.
5. Write failing tests for `FrameClock`: reported time is used verbatim; a `None`
   report falls back to the frame rate; a `None` frame rate falls back to the index;
   provenance is correct in all three cases; a regression is clamped, marked
   adjusted, and counted; index increments by exactly one per stamp.
6. Implement `FrameClock` until those tests pass, then wire it into the playback
   loop so every decoded frame produces a stamp, sampling `CAP_PROP_POS_MSEC` and
   observing which side of `read()` yields the frame just decoded on this backend.
7. Add `TrackId`, `TrackState`, `Track`, and the three parameter constants, with a
   test asserting `TrackId` displays as `#42` and that ids compare and sort.
8. Log the frame count, final media time, provenance mix, and adjustment count once
   at the end of the run.
9. Run the full validation suite and review the diff for scope.

## Validation

- Unit: `BBox` geometry and IoU parity with VE-004's existing values; `FrameClock`
  resolution order, provenance, monotonicity repair, and counting; `TrackId`
  formatting and ordering.
- Regression: the entire existing detector and CLI suite passes unmodified in
  behavior after the `Detection` change.
- Manual: run the release binary against `samples/test.mp4` and confirm the
  end-of-run log reports a plausible media duration with `Reported` provenance, then
  repeat against a source whose timestamps are unavailable to confirm the fallback.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx
```

## Handoff

Report the backend observed, which read ordering yields the decoded frame's own
timestamp, the provenance mix and adjustment count for the sample video, and
confirmation that VE-004's detection assertions are unchanged. Note any call site
where the `BBox` migration altered behavior rather than only syntax.
