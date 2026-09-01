# VE-002: Video decoding and playback

**Status:** `DONE` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-001  
**Implementation plan:** [`../plans/VE-002-video-decoding-and-playback-plan.md`](../plans/VE-002-video-decoding-and-playback-plan.md)

## Purpose

Turn the validated video path into the first visible vertical slice: sequential
OpenCV decoding with live playback and measurable decode/FPS output. This remains
a single-threaded baseline on which inference can be added without first building
a pipeline architecture.

## Requirements

### Video source

- Open the validated video path with OpenCV's file-backed `VideoCapture`.
- Fail before creating the playback loop if OpenCV cannot open the file, with an
  error that includes the video path.
- Decode one frame at a time on the calling thread.
- A failed read is an error; a successful read that produces an empty frame marks
  normal end-of-file.
- Preserve the decoded frame's native dimensions and color representation.

### Live playback

- Display each decoded frame in one named OpenCV window.
- Process frames as quickly as the decoder and display allow; do not pace playback
  to the source video's recorded frame rate.
- Poll UI events with the minimum practical wait and stop cleanly when the user
  presses Q, q, or Escape.
- Reaching end-of-file also exits successfully.
- Ensure the window is destroyed on normal EOF, user exit, and runtime error.

### Baseline measurements

- Measure decode latency around only the frame-read operation using a monotonic
  clock.
- Calculate processing FPS over a rolling interval of at least one second rather
  than using the source metadata FPS.
- Overlay the latest decode latency in milliseconds and rolling processing FPS on
  every displayed frame once those values are available.
- Metrics must describe actual observed work and must not claim inference timing
  before VE-003 adds inference.

## Constraints and non-goals

- No inference, detections, model inspection, bounding boxes, output encoding,
  source-rate pacing, frame queue, worker thread, batching, or dropped-frame policy.
- No camera devices or network streams.
- Keep the loop direct and synchronous; Milestone 3 owns pipeline parallelism.

## Acceptance criteria

1. A valid local video opens and displays decoded frames continuously.
2. The overlay shows measured decode milliseconds and processing FPS.
3. Q, q, Escape, and end-of-file all produce a successful clean shutdown.
4. An unreadable or unsupported video produces a contextual error without a
   panic or abandoned display window.
5. FPS is derived from completed loop iterations over elapsed monotonic time, not
   copied from video metadata.
6. Automated timing tests use a controllable calculation boundary and do not
   depend on wall-clock sleeps or a graphical display.
7. Formatting, linting, tests, and the release build pass.
