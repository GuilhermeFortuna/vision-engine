# AGENTS.md

## Project

**Vision Engine** is a high-performance, local-first visual intelligence platform written primarily in Rust.

Its long-term purpose is to turn cameras, videos, and images into structured, searchable events using computer vision, object tracking, GPU acceleration, and eventually custom neural networks.

Read `PROJECT.md` before making architectural or scope decisions.

---

## Current Development Phase

The project is at **Milestone 1 — Video + Object Detection**.

Development must proceed in small vertical checkpoints.

### Current checkpoint

Implement only the first video-ingestion slice:

```text
CLI video path
    ↓
OpenCV VideoCapture
    ↓
decode frames
    ↓
display frames
    ↓
basic timing / source metadata
```

Do **not** implement object detection yet unless the task explicitly says to move to the next checkpoint.

---

## Core Principles

- Build vertically.
- Keep each task small, bounded, visible, and testable.
- Prefer the simplest correct implementation.
- Do not introduce abstractions before they are needed.
- Do not optimize before the synchronous CPU pipeline is correct and measurable.
- Avoid speculative architecture.
- Measure performance changes instead of assuming they help.
- Preserve a clear path from raw video input to structured visual events.

---

## Primary Stack

### Language

- Rust
- Cargo

### Initial runtime libraries

- `anyhow`
- `tracing`
- `tracing-subscriber`

### Computer vision

- OpenCV for:
  - video capture
  - decoding
  - image manipulation
  - rendering
  - display

### Planned inference stack

Do not add until explicitly requested:

- ONNX Runtime through `ort`
- YOLO-family model exported to ONNX
- `ndarray`

### Planned concurrency

Do not add until justified by a later milestone:

- `rayon`
- native Rust threads
- `tokio`

### Planned storage

Do not add until persistence work begins:

- SQLite

---

## Repository Structure

Keep the repository simple for now:

```text
vision-engine/
├── src/
│   └── main.rs
├── models/
├── samples/
├── Cargo.toml
├── PROJECT.md
└── AGENTS.md
```

Do not convert this project into a multi-crate workspace unless the codebase has clearly grown enough to justify it.

---

## Scope Rules

### Allowed in the current checkpoint

- Accept a video path from the CLI.
- Open a local video file with OpenCV.
- Read frames sequentially.
- Display decoded frames.
- Read source metadata such as:
  - width
  - height
  - source FPS
- Measure useful timing such as:
  - per-frame decode latency
  - effective processing FPS
- Handle basic failures cleanly.

### Explicitly out of scope

Do not add any of the following unless the task explicitly requests it:

- YOLO inference
- ONNX Runtime
- object tracking
- persistence
- SQLite
- GPU acceleration
- ROCm
- async processing
- Tokio
- Rayon
- worker pools
- frame queues
- channels
- multi-camera support
- webcam support
- RTSP
- event engine
- semantic search
- custom model training
- web UI
- desktop UI
- authentication
- cloud infrastructure
- Kubernetes
- distributed workers
- plugin systems

---

## Agent Workflow

For every implementation task:

1. Read:
   - `AGENTS.md`
   - `PROJECT.md`
   - any task-specific spec or plan

2. Inspect the existing code before modifying it.

3. Restate internally the exact task boundary.

4. Implement only what is necessary for the requested checkpoint.

5. Do not silently expand scope.

6. Run all relevant validation commands.

7. Fix validation failures caused by the change.

8. Review the diff before finishing.

9. Report:
   - what changed
   - validation performed
   - any known limitation or follow-up

---

## Rust Commands

Unless a task says otherwise, run these commands from the **repository root**.

### Fast compile check

```bash
cargo check
```

### Formatting

```bash
cargo fmt --check
```

If formatting fails:

```bash
cargo fmt
```

### Linting

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Tests

```bash
cargo test
```

### Release build

```bash
cargo build --release
```

For changes affecting the executable path, a successful release build is required before completion.

---

## Validation Requirements

Before considering an implementation task complete, run at minimum:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```

If any command cannot be run because of an environment or external dependency issue:

- do not hide the failure
- report the exact blocker
- distinguish environment failures from code failures

---

## Error Handling

- Prefer `anyhow::Result` at application boundaries.
- Return useful errors rather than panicking.
- Do not use `unwrap()` or `expect()` in normal runtime paths unless an invariant is truly guaranteed and the reason is obvious.
- Include enough context for failures to be actionable.

Examples of errors that should be handled cleanly:

- missing CLI argument
- nonexistent video file
- OpenCV failing to open the source
- decode failure
- invalid or unavailable source metadata

---

## Logging and Metrics

Use `tracing` for diagnostic output when appropriate.

Prefer measurable values over vague performance claims.

Useful metrics for the initial pipeline include:

- source resolution
- source FPS
- frame count
- decode latency
- effective FPS

Do not introduce a complex telemetry system.

---

## Performance Guidance

Performance matters, but correctness comes first.

For now:

- keep the pipeline synchronous
- avoid unnecessary frame copies when reasonably simple
- do not add unsafe code for performance without evidence
- do not add concurrency merely because the system may need it later
- do not add GPU acceleration before a working CPU baseline exists

Later optimizations must be benchmarked.

---

## Architecture Guidance

Do not prematurely create modules, traits, services, factories, interfaces, queues, or generic abstractions for code that currently has only one implementation.

A small `main.rs` is acceptable during the first checkpoint.

Refactor only when the implementation has enough real complexity to justify it.

The intended long-term pipeline is:

```text
Decoder
   ↓
Frame Queue
   ↓
Preprocessor
   ↓
Inference
   ↓
Tracker
   ↓
Event Engine
   ├── Storage
   ├── Rules
   ├── Notifications
   └── Renderer
```

This is a direction, not a requirement for the initial implementation.

---

## Dependency Policy

- Prefer well-maintained crates with clear justification.
- Avoid adding dependencies for functionality easily handled by the standard library.
- Do not add planned future dependencies ahead of the milestone that needs them.
- Keep dependency versions intentional.
- Explain any non-obvious dependency addition in the task summary.

---

## Code Quality

Prefer:

- idiomatic Rust
- small functions
- explicit ownership
- clear error propagation
- simple control flow
- descriptive names
- minimal mutable state

Avoid:

- unnecessary cloning
- premature generics
- excessive trait abstraction
- large dependency trees without need
- hidden global state
- unsafe code without strong justification

---

## Git Discipline

Keep commits focused on one logical change.

Do not mix:

- refactors
- dependency upgrades
- formatting unrelated files
- unrelated cleanup

with a feature unless required for that feature.

Do not rewrite unrelated user changes.

---

## First Checkpoint Definition of Done

The initial video-ingestion checkpoint is complete when:

- the project builds in release mode
- a local video path can be supplied from the CLI
- OpenCV opens the video
- frames decode continuously until EOF
- decoded frames are displayed
- source resolution is reported
- source FPS is reported
- decode timing is reported
- effective FPS is reported
- missing/invalid input fails cleanly
- formatting, linting, tests, and release build pass

Example invocation:

```bash
cargo run --release -- samples/test.mp4
```

---

## Next Checkpoint

Only after the video-ingestion checkpoint is complete should the project move to model inference.

The next bounded task should be:

```text
Load a YOLO ONNX model using `ort`, inspect and validate the model
input/output tensors, and exit cleanly.

Do not integrate inference into the video loop yet.
```

Then:

```text
single image
    ↓
preprocess
    ↓
YOLO inference
    ↓
decode detections
    ↓
NMS
    ↓
render boxes
```

Only after single-image inference works should detection be integrated into the video loop.

---

## Final Rule

When uncertain between:

- implementing something now, or
- leaving it for a later milestone

default to **leaving it for later** unless it is necessary to complete the current vertical slice.
