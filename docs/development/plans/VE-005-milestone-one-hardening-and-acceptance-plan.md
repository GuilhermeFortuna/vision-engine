# VE-005 implementation plan: Milestone 1 hardening and acceptance

**Status:** `READY` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-005-milestone-one-hardening-and-acceptance-spec.md`](../specs/VE-005-milestone-one-hardening-and-acceptance-spec.md)  
**Depends on:** VE-004

## Current-system context

VE-001 through VE-004 establish the complete synchronous pipeline: validated local
paths, OpenCV file decoding and display, fixed CPU YOLOv8 inference, detection
post-processing, labeled boxes, performance overlays, and interactive shutdown.
This pair assumes those features exist and changes only what evidence or targeted
fixes show is needed for the Milestone 1 definition of done.

## Implementation decisions

- Audit each fallible boundary in execution order and add `anyhow::Context` at the
  narrow call site that knows the operation, path, or expected model contract.
  Preserve underlying OpenCV and ONNX Runtime errors in the chain.
- Keep one top-level owner for capture, detector session, frame, and display
  lifecycle. Do not add a service container or staged pipeline to manage cleanup.
- Use an explicit display guard or equivalent small scoped mechanism only if the
  existing control flow cannot guarantee cleanup on every post-window return path.
- Prefer buffer reuse supported by the existing OpenCV/ndarray/`ort` APIs, but do
  not add unsafe storage reuse or preallocation without evidence that allocation
  is the cause of sustained growth.
- Treat memory as an observed process property. Warm for two minutes, then sample
  RSS once per minute for at least ten more minutes. Account for bounded allocator
  high-water behavior separately from a sequence that continues rising with frame
  count.
- Do not add a benchmark framework for one acceptance run. Record hardware, build,
  model, input resolution, duration, metric ranges, and RSS observations in the
  implementation handoff.
- Keep tests headless by testing pure functions and startup boundaries. Reserve
  live window, model-quality, and long-run checks for explicit local smoke tests.

## Ordered implementation

1. Inventory all startup, per-frame, shutdown, and cleanup return paths from the
   completed VE-004 implementation.
2. Add missing contextual errors and make success/non-success exit behavior
   consistent.
3. Close any demonstrated cleanup or per-frame retention issue with the smallest
   local change, adding regression coverage where deterministic.
4. Run unit/integration coverage for argument, file, model-contract, transform,
   post-processing, and metric behavior.
5. Run the full required Rust validation suite and release build.
6. Exercise the real release pipeline and every manual exit/failure scenario.
7. Perform and record the warm-up plus sustained RSS observation.
8. Review the complete Milestone 1 diff for later-milestone scope and update
   `STATUS.md` only after all required evidence passes.

## Validation

- Automated cases: argument failures, missing/non-file paths, unsupported model
  contract, preprocessing geometry, NMS/filtering, metric calculation, and error
  exit codes that do not require GUI interaction.
- Manual failure cases: undecodable video, invalid ONNX content, OpenCV display
  failure where the environment permits, and clean Q/q/Escape plus EOF shutdown.
- Visual case: representative COCO objects at center and frame edges, checking box
  alignment, class names, confidence labels, and metric overlay readability.
- Sustained case: release build, two-minute warm-up, at least ten measured minutes,
  RSS sampled every minute with frame count and elapsed time.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/video.mp4 --model models/yolov8n.onnx
```

## Handoff

Report every validation command and manual scenario as pass, fail, or blocked.
Include release hardware, model identity and tensor contract, video resolution and
duration, decode/inference/FPS observations, post-warm-up/peak/final RSS, and any
remaining limitation. Mark VE-005 and Milestone 1 complete only when all acceptance
criteria have evidence; do not begin tracking or another milestone automatically.
