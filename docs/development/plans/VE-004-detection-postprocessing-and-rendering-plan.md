# VE-004 implementation plan: Detection post-processing and rendering

**Status:** `DONE` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-004-detection-postprocessing-and-rendering-spec.md`](../specs/VE-004-detection-postprocessing-and-rendering-spec.md)  
**Depends on:** VE-003

## Current-system context

VE-003 owns one CPU ONNX session, converts decoded frames to fixed YOLOv8 input,
and returns the `[1, 84, 8400]` FP32 output with letterbox transform metadata. The
video loop already renders frames and its three performance metrics. Raw output is
validated but deliberately uninterpreted.

## Implementation decisions

- Keep post-processing beside the concrete detector introduced by VE-003. Add one
  small `Detection` struct, not a public hierarchy or detector trait.
- Traverse the output by prediction index, reading channels 0-3 as `xywh` and
  channels 4-83 as COCO class scores. Reject any candidate whose required values
  are not finite.
- Use confidence `>= 0.25`. Convert to input-space corners, subtract left/top
  padding, divide by scale, clamp to `[0, frame width/height]`, then reject zero-area
  results.
- Sort candidates by descending confidence with a total deterministic fallback on
  class ID and coordinates. Avoid partial-comparison panics.
- Implement IoU and greedy class-aware NMS directly over the filtered vector at
  threshold `0.70`. The expected candidate count does not justify spatial indexes
  or another dependency.
- Store the canonical 80 COCO labels as a static ordered array and treat an
  out-of-range class ID as an invalid model output.
- Render boxes and labels using OpenCV `rectangle`, `get_text_size`, and `put_text`.
  Clamp the label background/text origin to the frame and reserve the upper-left
  metric area by moving a colliding detection label below its box when possible.
- Convert floating-point boxes to integer drawing coordinates only at the rendering
  boundary so post-processing tests retain precise geometry.

## Ordered implementation

1. Add `Detection`, the COCO labels, and raw-candidate extraction.
2. Add inverse-letterbox coordinate conversion, clamping, and invalid-box removal.
3. Implement deterministic sorting, IoU, and class-aware NMS.
4. Return retained detections from the existing per-frame detector call.
5. Add bounding-box and label rendering without changing the existing metric
   calculation or shutdown behavior.
6. Add pure unit tests for post-processing and rendering-boundary calculations.
7. Smoke-test detection alignment and class labels on representative local video.

## Validation

- Test candidates immediately below and at the confidence threshold.
- Test identical, disjoint, partially overlapping, cross-class, zero-area, and
  non-finite boxes.
- Test inverse transforms for landscape, portrait, square, padding boundaries, and
  coordinates extending beyond the frame.
- Test COCO class IDs 0 and 79 plus rejection of an out-of-range ID.
- Compare real-video box alignment on objects near the center and frame edges.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/video.mp4 --model models/yolov8n.onnx
```

## Handoff

Report unit-test coverage for filtering/NMS/coordinate restoration, the number and
classes of detections observed in the smoke test, visual alignment at padded and
frame edges, displayed performance values, and the release-build result.
