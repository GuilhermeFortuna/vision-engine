
| ID                                                                                                                                              | Batch | Status  | Depends on | Deliverable                                                       |
| ----------------------------------------------------------------------------------------------------------------------------------------------- | ----- | ------- | ---------- | ----------------------------------------------------------------- |
| [VE-001](specs/VE-001-cli-and-runtime-foundation-spec.md) / [Plan](plans/VE-001-cli-and-runtime-foundation-plan.md)                             | 01    | `DONE`  | None       | Validated CLI, local input contract, logging, and clean errors    |
| [VE-002](specs/VE-002-video-decoding-and-playback-spec.md) / [Plan](plans/VE-002-video-decoding-and-playback-plan.md)                           | 01    | `DONE`  | VE-001     | OpenCV video decoding, live playback, decode timing, and FPS      |
| [VE-003](specs/VE-003-yolov8-onnx-inference-spec.md) / [Plan](plans/VE-003-yolov8-onnx-inference-plan.md)                                       | 01    | `DONE`  | VE-002     | CPU YOLOv8 ONNX preprocessing, inference, and latency measurement |
| [VE-004](specs/VE-004-detection-postprocessing-and-rendering-spec.md) / [Plan](plans/VE-004-detection-postprocessing-and-rendering-plan.md)     | 01    | `DONE`  | VE-003     | Filtered detections, class-aware NMS, and labeled bounding boxes  |
| [VE-005](specs/VE-005-milestone-one-hardening-and-acceptance-spec.md) / [Plan](plans/VE-005-milestone-one-hardening-and-acceptance-plan.md)     | 01    | `DONE`  | VE-004     | Robust lifecycle and verified Milestone 1 acceptance              |
| [VE-006](specs/VE-006-tracking-domain-and-frame-timestamps-spec.md) / [Plan](plans/VE-006-tracking-domain-and-frame-timestamps-plan.md)         | 02    | `DONE`  | VE-005     | Frame stamps, shared bounding box, and track domain model         |
| [VE-007](specs/VE-007-kalman-motion-model-spec.md) / [Plan](plans/VE-007-kalman-motion-model-plan.md)                                           | 02    | `DONE`  | VE-006     | Constant-velocity Kalman prediction and recoverable updates       |
| [VE-008](specs/VE-008-detection-track-association-spec.md) / [Plan](plans/VE-008-detection-track-association-plan.md)                           | 02    | `DONE`  | VE-007     | Class-partitioned IoU cost and deterministic optimal assignment   |
| [VE-009](specs/VE-009-tracker-lifecycle-and-integration-spec.md) / [Plan](plans/VE-009-tracker-lifecycle-and-integration-plan.md)               | 02    | `DONE`  | VE-008     | Stable track IDs, lifecycle rules, and bounded tracker state      |
| [VE-010](specs/VE-010-tracking-visualization-and-metrics-spec.md) / [Plan](plans/VE-010-tracking-visualization-and-metrics-plan.md)             | 02    | `DONE`  | VE-009     | Rendering extraction, ID labels, and tracking metrics overlay     |
| [VE-011](specs/VE-011-milestone-two-hardening-and-acceptance-spec.md) / [Plan](plans/VE-011-milestone-two-hardening-and-acceptance-plan.md)     | 02    | `DONE`  | VE-010     | Deterministic ID evidence and verified Milestone 2 acceptance     |
| [VE-012](specs/VE-012-pipeline-stage-extraction-spec.md) / [Plan](plans/VE-012-pipeline-stage-extraction-plan.md)                               | 03    | `DONE`  | VE-011     | Stage modules, serial execution preserved, and a locked baseline  |
| [VE-013](specs/VE-013-frame-messages-and-bounded-queues-spec.md) / [Plan](plans/VE-013-frame-messages-and-bounded-queues-plan.md)               | 03    | `DONE`  | VE-012     | Concrete frame messages and a blocking bounded queue              |
| [VE-014](specs/VE-014-threaded-pipeline-runtime-spec.md) / [Plan](plans/VE-014-threaded-pipeline-runtime-plan.md)                               | 03    | `DONE`  | VE-013     | Stage threads, backpressure, shutdown, and baseline parity        |
| [VE-015](specs/VE-015-pipeline-instrumentation-spec.md) / [Plan](plans/VE-015-pipeline-instrumentation-plan.md)                                 | 03    | `DONE`  | VE-014     | Per-stage latency, queue depth, and named bottleneck reporting    |
| [VE-016](specs/VE-016-milestone-three-hardening-and-acceptance-spec.md) / [Plan](plans/VE-016-milestone-three-hardening-and-acceptance-plan.md) | 03    | `READY` | VE-015     | Verified parity, measured speedup, and Milestone 3 acceptance     |


Batch 02 implements Milestone 2 (object tracking) with a full SORT tracker:
class-partitioned association, a constant-velocity Kalman motion model, and a
media-time-based track lifecycle. The sustained acceptance run is driven by
`scripts/sustained-run.sh`, which loops the input in-process via
`--loop-for-seconds` and writes a per-sample CSV.

Batch 03 implements Milestone 3 (pipeline refactor). Execution splits into decode,
preprocess, inference, and tracking threads with rendering on the main thread,
connected by bounded blocking queues. Frames are never dropped: a full queue stalls
its producer, which keeps output frame-for-frame identical to the serial baseline
VE-012 locks in. That baseline, held in `docs/development/baselines/VE-012/`, is the
reference every later pair in the batch is measured against.