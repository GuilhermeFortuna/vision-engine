# VE-005: Milestone 1 hardening and acceptance

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-004  
**Implementation plan:** [`../plans/VE-005-milestone-one-hardening-and-acceptance-plan.md`](../plans/VE-005-milestone-one-hardening-and-acceptance-plan.md)

## Purpose

Turn the functional detection loop into the completed Milestone 1 baseline. This
pair closes failure-path and lifecycle gaps, verifies the full release executable
with real local assets, and records measured behavior without starting any later
milestone.

## Requirements

### Failure behavior

- Missing or non-file video/model paths retain VE-001's role-specific errors.
- A file OpenCV cannot decode, a model ONNX Runtime cannot load, and a model with
  the wrong tensor contract each fail with distinct actionable context.
- Frame decoding, preprocessing, inference, post-processing, rendering, event
  polling, and window cleanup failures preserve the operation that failed.
- Runtime failures return non-zero. Help, EOF, and Q/q/Escape return success.
- Expected user or environment errors do not panic and do not expose a Rust
  backtrace unless the user explicitly enables standard backtrace behavior.

### Resource lifecycle

- Load one model session and open one video capture/window per process run.
- Reuse the frame and long-lived detector state across iterations where supported.
- Per-frame tensors, raw outputs, and detections must become reclaimable after the
  frame is processed; no per-frame collection is retained across the video.
- Attempt display-window cleanup after normal completion and after any error that
  occurs once the window exists.
- If primary processing and cleanup both fail, report the processing failure while
  retaining cleanup information as additional context.

### Measured acceptance run

- Build and run the release executable with a local YOLOv8n COCO ONNX model and a
  representative local video containing recognizable COCO objects.
- Visually confirm aligned boxes, correct labels, readable confidence values, and
  simultaneous decode-latency, inference-latency, and processing-FPS overlays.
- Exercise normal EOF, Q/q or Escape exit, missing video, unreadable video,
  missing model, invalid ONNX, and unsupported tensor-shape paths.
- Run continuously for at least ten minutes after a two-minute warm-up using a
  sufficiently long or externally repeated input, sampling resident set size at
  least once per minute.
- Report the release hardware, video resolution, observed decode/inference/FPS
  values, starting post-warm-up RSS, peak RSS, and final RSS. Success requires no
  sustained monotonic growth attributable to retained per-frame state.

### Milestone boundary

- The final executable remains the direct synchronous CPU pipeline established by
  VE-001 through VE-004.
- Completing this pair makes Milestone 1 eligible to be marked complete; it does
  not authorize Milestone 2 work.

## Constraints and non-goals

- No tracking, persistence, camera/RTSP input, GPU execution, output encoding,
  frame queues, async runtime, parallel workers, model downloads, additional model
  families, benchmark harness, or general refactor.
- Do not claim leak freedom solely from Rust ownership or a short run; use the
  sustained-run observation required above.
- Do not commit proprietary or large model/video assets to satisfy validation.

## Acceptance criteria

1. All failure cases return the expected success/non-success status with contextual
   messages and no panic.
2. The live release executable detects recognizable objects with visually aligned,
   correctly labeled boxes and all required metrics.
3. EOF and interactive exit release the window and terminate promptly.
4. The sustained run records the required RSS samples and shows no continuing
   per-frame memory growth after warm-up.
5. Unit and integration tests cover all deterministic logic and practical startup
   failures without requiring a graphical session or external network access.
6. `cargo fmt --check`, strict Clippy, tests, and the release build all pass.
7. Any unavailable model, video, display, or system runtime is reported as an exact
   environment blocker; blocked evidence is never represented as a passing check.
8. The final diff contains no functionality assigned to Milestone 2 or later.
