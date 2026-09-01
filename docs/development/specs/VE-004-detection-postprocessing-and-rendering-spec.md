# VE-004: Detection post-processing and rendering

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-003  
**Implementation plan:** [`../plans/VE-004-detection-postprocessing-and-rendering-plan.md`](../plans/VE-004-detection-postprocessing-and-rendering-plan.md)

## Purpose

Convert VE-003's raw YOLOv8 output into trustworthy detections and render those
detections on the original video frames. This completes the functional Milestone
1 pipeline from video pixels to visible labeled objects.

## Requirements

### Detection model

- Define one concrete detection value containing a zero-based COCO class ID,
  confidence, and an axis-aligned source-frame rectangle in floating-point
  coordinates.
- Interpret each raw YOLOv8 prediction as center-x, center-y, width, height, and 80
  class scores.
- Select the highest class score for each candidate. The selected score is the
  detection confidence; no separate objectness term is present in this export.
- Discard non-finite coordinates or scores, non-positive boxes, and candidates
  below confidence `0.25`.

### Coordinate restoration and suppression

- Convert retained boxes from center-width-height to corner coordinates.
- Reverse VE-003's letterbox padding and scale to recover source-frame coordinates.
- Clamp final corners to the source frame and discard boxes that become empty.
- Apply greedy, descending-confidence non-maximum suppression independently per
  class with IoU threshold `0.70`.
- Use stable deterministic ordering for equal-confidence candidates so tests and
  rendered output are repeatable.

### Labels and rendering

- Map class IDs through a fixed, ordered list of the 80 COCO labels corresponding
  to the supported YOLOv8n model.
- Draw a visible rectangle for every retained detection on the original frame.
- Draw a label containing the class name and confidence rounded to two decimal
  places, positioned within the visible frame even near its top or side edges.
- Retain the decode latency, inference latency, and processing FPS overlay from
  earlier pairs without allowing detection labels to erase it.
- The display continues to support EOF and Q/q/Escape shutdown behavior.

## Constraints and non-goals

- No tracking IDs, temporal smoothing, zones, events, segmentation, alternate
  label sets, user-configurable thresholds, output video encoding, or model-family
  abstraction.
- Do not use unsafe code or add an NMS dependency for the small fixed algorithm.
- Do not parallelize post-processing or rendering without measurement.

## Acceptance criteria

1. Supported-model output produces correctly classified, confidence-filtered
   detections in source-frame coordinates.
2. Class-aware NMS suppresses overlapping boxes of the same class but retains
   overlapping boxes of different classes.
3. Boxes remain within frame bounds after reversing letterbox geometry, including
   portrait, landscape, square, and edge-touching cases.
4. Every retained detection is rendered with the correct COCO name and confidence.
5. A real-video smoke test visibly aligns boxes with objects and retains all three
   performance metrics.
6. Pure unit tests cover coordinate conversion, IoU, filtering, NMS, deterministic
   ordering, label lookup, and invalid numeric values.
7. Formatting, linting, tests, and the release build pass.
