# VE-003: YOLOv8 ONNX inference

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-002  
**Implementation plan:** [`../plans/VE-003-yolov8-onnx-inference-plan.md`](../plans/VE-003-yolov8-onnx-inference-plan.md)

## Purpose

Add the first real visual-intelligence workload to the synchronous video loop:
preprocess every decoded frame and execute a fixed YOLOv8n COCO ONNX model on the
CPU. This pair establishes a correct, measurable inference baseline and exposes
raw predictions for VE-004 without generalizing to other model families.

## Requirements

### Supported model contract

- Support an Ultralytics YOLOv8n object-detection model exported to ONNX as FP32,
  fixed batch size 1, fixed `640x640` image size, and without embedded NMS.
- The model has one FP32 input shaped `[1, 3, 640, 640]` and one FP32 detection
  output shaped `[1, 84, 8400]`.
- Load the model path selected by VE-001 once before opening the frame loop.
- Enable ONNX Runtime graph optimization while using CPU execution only.
- Validate the model's input/output count, element type, and dimensions during
  startup. A mismatch is an actionable unsupported-model error, not a later index
  failure or panic.

### Frame preprocessing

- Resize each decoded frame to fit within `640x640` while preserving aspect ratio.
- Center the resized image on a `640x640` canvas padded with pixel value 114 on all
  color channels.
- Convert OpenCV BGR pixels to RGB.
- Convert pixels to FP32, normalize channel values to `[0.0, 1.0]`, and arrange
  them as contiguous NCHW data shaped `[1, 3, 640, 640]`.
- Retain the exact scale and horizontal/vertical padding used for each frame so
  VE-004 can map predictions back to source coordinates.
- Reject empty or unsupported input frames with context rather than indexing
  invalid memory.

### Inference and timing

- Run one inference for every successfully decoded frame on the calling thread.
- Measure inference latency around the ONNX Runtime `run` call only. Preprocessing
  remains outside that measurement.
- Extract and validate the raw FP32 output without interpreting detections in this
  pair.
- Extend the overlay with the latest inference latency in milliseconds while
  retaining decode latency and processing FPS.
- Reuse the loaded session for the entire video. Do not recreate it per frame.

### Local operation

- Use `ort` pinned to `=2.0.0-rc.13` because its 2.0 API is not yet stable, plus
  the compatible `ndarray` 0.17 release.
- Build against ONNX Runtime's CPU path only; do not enable CUDA, ROCm, MIGraphX,
  or other accelerator execution-provider features.
- Runtime inference performs no network access. The application neither downloads
  a model nor invokes Python or the Ultralytics package.

## Constraints and non-goals

- No output decoding, confidence filtering, NMS, boxes, labels, batching, dynamic
  model shapes, alternate input sizes, segmentation, pose, GPU support, or generic
  detector interface.
- No attempt to support multiple YOLO export layouts.
- No optimization based on unsafe code, preallocated ONNX outputs, or custom
  allocators without a measured baseline.

## Acceptance criteria

1. A supported YOLOv8n ONNX model loads once and performs CPU inference for every
   decoded frame.
2. Preprocessing produces the required RGB, normalized FP32 NCHW tensor and
   correct scale/padding metadata for landscape, portrait, and square frames.
3. Unsupported input/output type, rank, dimension, or count fails at startup with
   the expected contract in the error.
4. The display reports decode latency, inference latency, and processing FPS as
   separate measured values.
5. A deterministic preprocessing test verifies channel order, normalization,
   dimensions, padding, and transform metadata.
6. Session reuse is visible in control flow and no per-frame model load occurs.
7. Formatting, linting, tests, and the release build pass.
