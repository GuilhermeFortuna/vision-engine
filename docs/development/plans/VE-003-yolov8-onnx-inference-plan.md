# VE-003 implementation plan: YOLOv8 ONNX inference

**Status:** `DONE` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-003-yolov8-onnx-inference-spec.md`](../specs/VE-003-yolov8-onnx-inference-spec.md)  
**Depends on:** VE-002

## Implementation references

- `ort` 2.0.0-rc.13 API and features: <https://docs.rs/crate/ort/2.0.0-rc.13>
- Ultralytics ONNX export behavior: <https://docs.ultralytics.com/modes/export/>

## Current-system context

VE-002 produces one decoded BGR `Mat` at a time in a direct playback loop and
already measures decode/FPS. The model path is validated by VE-001 but unused.
There is no detection type, tensor code, or model module. The current structure is
still intentionally smaller than the staged pipeline reserved for Milestone 3.

## Implementation decisions

- Add exact `ort = "=2.0.0-rc.13"` and compatible `ndarray = "0.17"`
  dependencies. Keep `ort`'s standard CPU/runtime, ndarray, and tracing support;
  enable no accelerator feature.
- Introduce one concrete detector module only when inference lands. It owns the
  `Session`, preprocessing function, fixed model constants, raw-output validation,
  and per-frame transform metadata. Do not add a trait for its sole implementation.
- Build the `Session` once with level-3 graph optimization and commit it from the
  configured model file. Do not choose a thread count before measuring defaults.
- Inspect session inputs and outputs immediately after loading. Require exactly
  one input/output, FP32, and the fixed shapes from the Spec. Use the model's
  declared input name when constructing the run inputs rather than assuming a
  string at each frame.
- Use OpenCV operations for BGR-to-RGB conversion, aspect-preserving resize, and
  constant border padding. Copy the resulting bytes once into an ndarray-backed
  NCHW FP32 tensor using explicit channel indexing.
- Represent the inverse mapping as scale plus left/top padding. Compute dimensions
  deterministically, assigning odd leftover padding so left/top use integer floor
  and right/bottom receive the remainder.
- Return a narrow raw-inference result containing the validated prediction tensor,
  transform metadata, and measured inference duration. VE-004 will consume it.
- Measure only `Session::run` as inference time. Do not fold preprocessing,
  extraction, overlay, or display into the number labeled `inference`.

## Ordered implementation

1. Add and pin the inference dependencies and update the lockfile.
2. Add the concrete detector/session owner and supported-model validation.
3. Implement letterbox, BGR-to-RGB, normalization, NCHW conversion, and transform
   metadata.
4. Run the session once per decoded frame and validate/extract the raw output.
5. Add inference latency to the existing metric overlay.
6. Add deterministic preprocessing and model-contract tests; use a small test ONNX
   fixture only if contract validation cannot be tested without one.
7. Smoke-test the full loop with a local YOLOv8n COCO ONNX model and video.

## Validation

- Test square, landscape, portrait, and odd-dimension letterboxing.
- Test known BGR pixels for RGB channel order and normalized NCHW positions.
- Test model-contract errors for the wrong type, rank, input size, or output shape
  when suitable fixtures can remain small and redistributable.
- Verify that the displayed inference duration excludes preprocessing.
- Run a real-model smoke test and report model load time separately from per-frame
  latency.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/video.mp4 --model models/yolov8n.onnx
```

## Handoff

Report the tested model's input/output names and shapes, model load time, observed
inference latency, preprocessing cases covered, and release-build result. Report
missing model/video/display assets as environment blockers without committing or
downloading them as an undocumented side effect.
