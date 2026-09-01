# VE-010 implementation plan: Tracking visualization and metrics

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-010-tracking-visualization-and-metrics-spec.md`](../specs/VE-010-tracking-visualization-and-metrics-spec.md)  
**Depends on:** VE-009

## Current-system context

`main.rs` carries argument parsing, startup validation, the playback loop,
`draw_detections`, and `draw_metrics_overlay`, and VE-009 has just added tracker
wiring to it. `draw_detections` already solves in-frame label placement and avoids
the metrics area. That behavior is worth preserving exactly; this pair moves it,
then retargets it from detections to tracks.

## Interfaces produced

```rust
// src/render.rs
pub struct FrameMetrics {
    pub decode_ms: f64,
    pub inference_ms: f64,
    pub tracking_ms: f64,
    pub fps: Option<f64>,
    pub confirmed_tracks: usize,
}

pub fn draw_tracks(frame: &mut Mat, tracks: &[Track]) -> Result<()>;
pub fn draw_metrics_overlay(frame: &mut Mat, metrics: &FrameMetrics) -> Result<()>;

// pure, unit-testable helpers
pub(crate) fn track_color(id: TrackId) -> Scalar;
pub(crate) fn label_text(track: &Track) -> String;
pub(crate) fn label_origin(
    box_left: i32, box_top: i32, box_bottom: i32,
    bg_w: i32, bg_h: i32, frame_w: i32, frame_h: i32,
) -> (i32, i32);
```

## Implementation decisions

- The extraction happens first and alone: move `draw_detections`,
  `draw_metrics_overlay`, and the drawing constants into `src/render.rs` with no
  behavioral change, run the suite, and commit before touching what is drawn. A
  pure move that is separately verified makes the later diff readable.
- While moving, lift the label-placement arithmetic out of the drawing loop into
  `label_origin`, which takes integers and returns a position. It is currently
  inline and untestable; as a free function it can be asserted at every edge without
  a display. `track_color` and `label_text` are extracted for the same reason.
- `draw_detections` becomes `draw_tracks` over `&[Track]`. The rectangle, label
  background, and text calls are unchanged; only the source of the box, the colour,
  and the label string differ.
- Label text: `"{class} {id} {confidence:.2}"` for `Confirmed`, and `"{class} ?"`
  — `TrackId`'s `Display` from VE-006 already supplies the `#`, so the format string
  must not add a second one. Assert the rendered form `person #42 0.91` in the test.
  for `Tentative` — no identity is shown, because presenting an id that may be
  discarded next frame would misrepresent stability. `Lost` never reaches the
  renderer, so `draw_tracks` skips it defensively rather than relying on the caller.
- Tentative styling: one-pixel box outline against two for confirmed, and no filled
  label background, so tentative boxes read as provisional without a second colour
  scheme. This keeps the distinction in weight rather than hue, leaving hue free to
  carry identity.
- `track_color` derives hue from the id by golden-ratio hashing —
  `hue = (id as f64 * 0.618_033_988_75).fract() * 360.0` — converted to BGR at fixed
  saturation and value. This gives well-separated colours for consecutive ids,
  depends only on the id so the colour is stable for the track's life, and needs no
  stored palette that would grow with track count.
- The metrics overlay gains two lines, `Tracking: N.N ms` and `Tracks: N`, drawn
  below the existing three at the established 30-pixel spacing. `METRICS_AREA_BOTTOM`
  rises from 100 to 160 to match, so `label_origin` keeps labels clear of the taller
  overlay.
- `tracking_ms` is the value VE-009 measured. `draw_metrics_overlay` neither times
  nor derives it. `confirmed_tracks` is counted from the slice being drawn, so the
  number on screen always describes the frame on screen.
- `main.rs` keeps the loop and now builds a `FrameMetrics` per iteration. Ordering is
  unchanged: tracks are drawn first, metrics last, so labels can never overwrite the
  overlay.

## Ordered implementation

1. Create `src/render.rs`, move both drawing functions and their constants verbatim,
   update `main.rs` to call them, run the full suite, and commit the pure move.
2. Extract `label_origin`, `track_color`, and `label_text` as free functions, keeping
   `label_origin`'s existing arithmetic identical. Run the suite and commit.
3. Write failing tests for `label_origin`: a box at the top edge places the label
   below it; at the bottom edge, above; at the right edge, shifted left to fit; a box
   in the metrics area is pushed clear of it; a label wider than the frame still
   yields a non-negative origin.
4. Fix any case the existing arithmetic gets wrong, keeping previously correct cases
   unchanged.
5. Write failing tests for `track_color`: the same id yields the same colour across
   repeated calls, consecutive ids differ measurably, and every channel stays within
   zero to 255.
6. Implement `track_color`.
7. Write failing tests for `label_text`: a confirmed track renders exactly
   `person #42 0.91`, with a single `#`; a tentative track renders no id.
8. Implement `label_text` and convert `draw_detections` into `draw_tracks`, applying
   the tentative styling and skipping `Lost`.
9. Add `FrameMetrics`, extend the overlay with the tracking and track-count lines,
   and raise `METRICS_AREA_BOTTOM` to 160.
10. Update `main.rs` to build `FrameMetrics` and call `draw_tracks`, keeping the
    draw-tracks-then-metrics order.
11. Run the release binary on the sample video and confirm visually: identities
    persist and hold colour, tentative boxes look provisional, labels stay in frame at
    every edge, and all five metrics are readable simultaneously.
12. Run the full validation suite.

## Validation

- Unit: `label_origin` at top, bottom, left, right, and corner positions plus the
  oversized-label case; `track_color` stability, separation, and range;
  `label_text` for both renderable states.
- Manual visual: sample video with objects reaching the frame edges, checking label
  containment, colour stability across frames, tentative versus confirmed styling,
  and overlay legibility with many tracks present.
- Regression: rendering behavior after step 1 is identical to before it.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx
```

## Handoff

Report `main.rs`'s line count before and after the extraction, any label-placement
case the existing arithmetic got wrong and how it was corrected, the colour
separation observed for consecutive ids, and confirmation that all five metrics
appear together with labels never overlapping the overlay.
