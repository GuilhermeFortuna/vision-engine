# Vision Engine

A high-performance, local-first visual intelligence platform written primarily in Rust.

The project turns cameras, videos, and images into structured, searchable events using computer vision, object tracking, GPU acceleration, and eventually custom neural networks.

## Vision

Build a reusable visual intelligence engine that can:

- Detect and track objects in real time.
- Convert video streams into structured events.
- Search historical footage by object, event, time, or semantic meaning.
- Trigger automations based on visual events.
- Run locally without depending on cloud inference.
- Scale from a single video file to multiple live camera streams.
- Leverage both high-core-count CPUs and modern GPUs.
- Support custom models and specialized computer vision workloads.

## Example Applications

### Smart Security

Examples:

- Detect people entering a driveway.
- Record clips only when relevant activity occurs.
- Trigger alerts for activity during specific time windows.
- Distinguish between known and unknown objects or people.
- Track movement across multiple cameras.

### Searchable Video

Convert recorded footage into structured data such as:

```text
14:03:17  person #42 entered
14:03:24  person #42 carrying backpack
14:07:51  red car #18 arrived
14:08:12  person #43 exited car #18
14:12:44  dog #7 entered
```

Future queries could include:

```text
Show every car detected between 14:00 and 16:00.
When did the dog leave?
Show every time a package was delivered.
Where was my backpack last seen?
```

### Object History

Maintain temporal knowledge about tracked objects:

```text
Object: backpack #12
First seen: 14:03
Last seen: 17:42
Last location: office chair
```

### Occupancy and Movement Analytics

Possible capabilities:

- Entrance and exit counts.
- Dwell time.
- Movement trajectories.
- Zone occupancy.
- Object flow between areas.
- Multi-camera tracking.

### Visual Automation

Example rule:

```text
WHEN person enters driveway
AND time BETWEEN 00:00 AND 06:00
THEN save_clip(30s)
AND notify()
```

### Custom Neural Networks

The system should eventually support:

```text
collect footage
    ↓
extract frames
    ↓
label data
    ↓
train model
    ↓
export model
    ↓
Rust inference engine
```

This allows specialized recognition for objects, environments, or workflows that generic models do not handle well.

---

# Core Architecture

```text
Cameras / Videos / Images
          │
          ▼
   ┌──────────────┐
   │ Vision Engine │
   └──────┬───────┘
          │
   ┌──────┴─────────────┐
   │                    │
Detection           Understanding
Tracking            OCR
Segmentation        Embeddings
Pose                Custom models
   │                    │
   └──────────┬─────────┘
              ▼
        Structured Events
              │
     ┌────────┼─────────┐
     ▼        ▼         ▼
   Search   Rules    Automation
```

## Processing Pipeline

The runtime should evolve toward a staged pipeline:

```text
Decoder
   │
   ▼
Frame Queue
   │
   ▼
Preprocessor
   │
   ▼
Inference
   │
   ▼
Tracker
   │
   ▼
Event Engine
   │
   ├── Storage
   ├── Rules
   ├── Notifications
   └── Renderer
```

Each stage should eventually be independently parallelizable.

---

# Primary Language

## Rust

Rust is the main implementation language because the project benefits from:

- High performance.
- Memory safety.
- Low-level control.
- Efficient multithreading.
- SIMD.
- Zero-copy processing.
- GPU integration.
- Async I/O.
- Predictable resource usage.
- Strong suitability for long-running local services.

The project should use Rust where performance, concurrency, and systems-level control matter.

---

# Initial Technology Stack

## Runtime

- Rust
- Cargo
- `anyhow`
- `serde`
- `serde_json`
- `tracing`
- `tracing-subscriber`

## Computer Vision

- OpenCV for initial capture, decoding, and rendering.
- ONNX Runtime through the Rust `ort` crate.
- YOLO-family object detection model exported to ONNX.
- `ndarray` for tensor manipulation.

## Parallelism

- `rayon`
- Rust threads
- `tokio` where asynchronous I/O becomes useful.

## Storage

Start with SQLite.

Potential future additions:

- PostgreSQL
- Vector database or vector extension
- Columnar analytics storage
- Object storage for clips and frames

## GPU

GPU acceleration should be introduced only after the CPU pipeline is correct.

Long-term target:

```text
ONNX Model
    ↓
Rust
    ↓
ONNX Runtime
    ↓
AMD-compatible execution backend
    ↓
ROCm / GPU runtime
    ↓
GPU
```

GPU acceleration should be treated as an optimization layer, not a prerequisite for initial correctness.

---

# Domain Model

The system should progressively move from raw pixels toward structured events.

## Detection

```rust
Detection {
    class: "person",
    confidence: 0.97,
    bbox: ...,
    timestamp: ...,
}
```

## Tracked Object

```rust
TrackedObject {
    id: 42,
    class: "person",
    position: ...,
    first_seen: ...,
    last_seen: ...,
}
```

## Events

Potential event types:

```rust
enum VisionEvent {
    ObjectAppeared(TrackedObject),
    ObjectDisappeared(TrackedObject),
    ObjectEnteredZone {
        object: TrackedObject,
        zone: ZoneId,
    },
    ObjectExitedZone {
        object: TrackedObject,
        zone: ZoneId,
    },
}
```

Example output:

```text
person #42 entered driveway
car #7 appeared
dog #3 entered backyard
person #42 left driveway
```

---

# Multi-Camera Direction

A major long-term goal is to reason about physical space rather than isolated video streams.

```text
Camera A             Camera B
   │                    │
   └──── person #37 ────┘
             │
             ▼
        World Model
             │
       position / path
```

Possible future capabilities:

- Cross-camera identity association.
- World-coordinate mapping.
- Floor-plan visualization.
- Last-known-location queries.
- Trajectory history.
- Object handoff between cameras.
- Spatial event rules.

Example:

```text
Where is the dog?

Last observed in the backyard at 18:42.
```

---

# Development Milestones

## Milestone 1 — Video + Object Detection

Build the smallest complete vertical slice.

Input:

```bash
vision-engine video.mp4
```

Pipeline:

```text
MP4
 ↓
Decode Frames
 ↓
YOLO Inference
 ↓
Bounding Boxes
 ↓
Rendered Output
```

Requirements:

- Accept a video file.
- Decode frames.
- Run object detection.
- Draw labeled bounding boxes.
- Display FPS.
- Display inference latency.
- Use CPU inference initially.

Do not add tracking, persistence, cameras, or GPU acceleration yet.

## Milestone 2 — Object Tracking

Add stable IDs across frames.

```text
frame 1 → person #42
frame 2 → person #42
frame 3 → person #42
```

Requirements:

- Stable track IDs.
- Track lifetime.
- First-seen and last-seen timestamps.
- Track confidence.
- Track disappearance handling.

## Milestone 3 — Pipeline Refactor

Split the monolithic implementation into independent stages:

```text
decode
preprocess
infer
track
store
render
```

Introduce:

- Bounded queues.
- Backpressure.
- Parallel workers.
- Profiling.
- Per-stage timing.

## Milestone 4 — Event Engine

Convert tracked objects into events.

Examples:

- Object appeared.
- Object disappeared.
- Object entered zone.
- Object exited zone.
- Object remained in zone.
- Object crossed line.

## Milestone 5 — GPU Acceleration

Move inference and suitable preprocessing workloads to the GPU.

Goals:

- Benchmark CPU vs GPU.
- Reduce frame copies.
- Explore zero-copy paths.
- Profile end-to-end latency.
- Preserve a CPU fallback path.

## Milestone 6 — Camera Support

Add:

```bash
vision-engine camera /dev/video0
```

Then support network streams:

```bash
vision-engine rtsp://camera/stream
```

## Milestone 7 — Persistence

Persist:

- Sources
- Frames
- Detections
- Tracks
- Events
- Zones
- Clips
- Model metadata

Example query:

```sql
SELECT *
FROM events
WHERE class = 'person'
  AND event_type = 'entered_zone'
  AND zone = 'driveway';
```

## Milestone 8 — Search

Add structured search first.

Then add semantic search.

Potential examples:

```text
Show every person detected yesterday.
Show every car that entered the driveway.
When was this backpack last seen?
Show clips where someone carried a package.
```

## Milestone 9 — Advanced Vision

Potential additions:

- Semantic segmentation.
- Pose estimation.
- OCR.
- Face embeddings.
- Object embeddings.
- Re-identification.
- Optical flow.
- Depth estimation.
- Custom classifiers.
- Trajectory prediction.

## Milestone 10 — Multi-Camera World Model

Build a spatial-temporal model across camera sources.

Potential capabilities:

- Cross-camera handoff.
- Floor-plan mapping.
- Last-known position.
- Movement histories.
- Multi-room tracking.
- Scene graphs.

---

# Performance Goals

Performance is a core feature of the project.

Areas to explore:

- SIMD.
- Frame batching.
- CPU affinity.
- Work stealing.
- Memory pools.
- Lock-free queues.
- Zero-copy buffers.
- GPU-resident frames.
- Hardware video decode.
- Hardware video encode.
- Async camera ingestion.
- Parallel model execution.
- Pipeline backpressure.
- Efficient clip extraction.

Every optimization should be benchmarked.

---

# Initial Repository Structure

Start simple:

```text
vision-engine/
├── src/
│   └── main.rs
├── models/
├── samples/
└── Cargo.toml
```

Do not introduce a multi-crate workspace prematurely.

When complexity justifies it, evolve toward:

```text
vision-engine/
├── crates/
│   ├── vision-core/
│   ├── vision-video/
│   ├── vision-inference/
│   ├── vision-tracking/
│   ├── vision-events/
│   └── vision-storage/
├── apps/
│   ├── vision-cli/
│   └── vision-viewer/
├── models/
├── samples/
├── benches/
└── Cargo.toml
```

---

# First Implementation Task

The first AI coding task should be:

> Build a Rust application that accepts a video file, decodes frames, performs YOLO ONNX object detection using `ort`, renders labeled bounding boxes, and displays decode/inference/FPS timing. Keep the architecture minimal. Use CPU inference initially; do not implement GPU acceleration, tracking, persistence, or camera support yet.

## Definition of Done

The first milestone is complete when:

- The project builds in release mode.
- A local video can be provided as input.
- Objects are detected correctly.
- Bounding boxes and labels are rendered.
- FPS is displayed.
- Inference latency is displayed.
- The system runs continuously without leaking memory.
- Basic errors are handled cleanly.

---

# Non-Goals for the Initial Version

Do not prematurely add:

- Web UI.
- Cloud infrastructure.
- Authentication.
- Distributed workers.
- Kubernetes.
- Multiple databases.
- Complex plugin systems.
- Custom model training.
- Multi-camera tracking.
- GPU-specific optimizations.
- Semantic search.

The first priority is to make the core visual pipeline work correctly and measurably fast.

---

# Project Principle

Build vertically.

Each milestone should produce something visible, measurable, and useful before expanding the architecture.

The long-term goal is not merely object detection.

It is a local system that can observe visual streams, build a structured temporal model of what happened, and make that information searchable and actionable.
