# VE-002 implementation plan: Video decoding and playback

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-002-video-decoding-and-playback-spec.md`](../specs/VE-002-video-decoding-and-playback-spec.md)  
**Depends on:** VE-001

## Current-system context

VE-001 leaves one synchronous application function with validated video and model
paths. OpenCV is already a dependency, but no backend is opened and the model path
is intentionally unused. This pair extends that application path rather than
creating the future staged pipeline described for Milestone 3.

## Implementation decisions

- Keep capture, timing, overlay, event polling, and cleanup in the current
  executable path. Extract only pure metric calculations when testing requires it.
- Convert the video path for OpenCV at the call boundary. If the installed OpenCV
  binding requires UTF-8, reject an unrepresentable path there with a contextual
  error rather than making CLI parsing lossy.
- Use `VideoCapture::from_file` with `CAP_ANY`, then verify `is_opened` before the
  loop.
- Reuse one `Mat` frame across reads. Interpret a successful empty frame as EOF;
  propagate a returned OpenCV read error.
- Use `Instant` around `read` for decode latency. Maintain a small accumulator of
  completed frames and elapsed time, refreshing the displayed processing FPS once
  at least one second has elapsed.
- Draw two text lines with `imgproc::put_text` before `highgui::imshow`: decode
  milliseconds and processing FPS. Use readable fixed styling without creating a
  rendering abstraction.
- Poll `highgui::wait_key(1)` and recognize Escape (`27`) plus case-insensitive Q.
- Route loop completion through explicit window cleanup. If processing and cleanup
  both fail, preserve the processing error and attach cleanup context.

## Ordered implementation

1. Open and verify the file-backed capture from VE-001's video path.
2. Create the named display window and the reusable frame.
3. Implement the sequential read/EOF loop with monotonic decode timing.
4. Add rolling FPS accumulation and the two-line overlay.
5. Display frames, poll exit keys, and centralize cleanup for every return path.
6. Add pure tests for FPS-window updates and key interpretation.
7. Smoke-test with a real local video in addition to the automated suite.

## Validation

- Test FPS aggregation before and after the one-second update boundary.
- Test Escape and upper/lowercase Q recognition independently of OpenCV input.
- Manually verify live playback, overlay readability, EOF, user exit, and an
  unsupported input file.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
target/release/vision-engine samples/video.mp4 --model models/yolov8n.onnx
```

## Handoff

Report the tested video properties, observed decode latency/FPS, each shutdown
path exercised, and the release-build result. If local assets or a display are
unavailable, identify those as environment blockers rather than treating the
manual smoke test as passed.
