<div align="center">

# 👁️ Vision Engine

**High-performance, local-first visual intelligence platform written in Rust.**

Turn cameras, videos, and images into structured, searchable events in real time.

[![Rust](https://img.shields.io/badge/rust-2024%20edition-orange?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-MIT%20%2F%20Apache--2.0-blue)](#license)
[![Status](https://img.shields.io/badge/status-active%20development-emerald)]()
[![Platform](https://img.shields.io/badge/platform-linux-lightgrey?logo=linux&logoColor=white)]()

</div>

---

## 🌟 Overview

**Vision Engine** is an edge-native, local-first visual computing platform. Rather than streaming raw video feeds to expensive and latency-prone cloud APIs, Vision Engine processes visual streams on-device to extract semantic meaning, track physical objects across space and time, and emit structured events for search and automation.

### Why Vision Engine?

- 🔒 **100% Local & Private:** Compute happens on your hardware. No cloud subscriptions, zero vendor lock-in, and full data privacy.
- ⚡ **Engineered for Speed:** Built from the ground up in Rust for zero-cost abstractions, predictable memory footprint, SIMD vectorization, and multi-core parallelism.
- 🧠 **Pixel to Event Pipeline:** Translates unstructured pixel streams into rich, queryable event streams (`ObjectAppeared`, `EnteredZone`, `DwellTimeExceeded`).
- 🌐 **Spatial-Temporal World Model:** Designed to scale from single video files to coordinated multi-camera networks with cross-camera tracking and floor-plan spatial awareness.
- 🚀 **Hardware Accelerated:** Built with a clean path from CPU execution to modern GPU acceleration (ROCm / CUDA / DirectML) via ONNX Runtime.

---

## 📐 Core Architecture & Pipeline

Vision Engine adopts a modular, staged pipeline design engineered for high-throughput, low-latency processing:

```text
Cameras / RTSP / Video Files
            │
            ▼
     ┌──────────────┐
     │ Video Decode │ (OpenCV / Hardware Decoders)
     └──────┬───────┘
            │ Bounded Queue (Zero-copy / Preprocessing)
            ▼
     ┌──────────────┐
     │  Inference   │ (ONNX Runtime / YOLO / Embeddings)
     └──────┬───────┘
            │ Detections (Class, Confidence, Bounding Box)
            ▼
     ┌──────────────┐
     │ Object Track │ (Stable IDs, Trajectories, State)
     └──────┬───────┘
            │ Tracked Objects
            ▼
     ┌──────────────┐
     │ Event Engine │ (Zone In/Out, Line Cross, Dwell Time)
     └──────┬───────┘
            │
    ┌───────┼──────────────────────┐
    ▼       ▼                      ▼
┌────────┐ ┌────────────────────┐ ┌───────────────┐
│ SQLite │ │ Search & Analytics │ │ Notifications │
│ Storage│ │ (Temporal/Semantic)│ │ & Automations │
└────────┘ └────────────────────┘ └───────────────┘
```

### Domain Abstraction

```rust
// 1. Raw Detection
Detection {
    class: "person",
    confidence: 0.97,
    bbox: Rect { x: 120, y: 80, w: 64, h: 180 },
    timestamp: 1714567200,
}

// 2. Continuous Tracked Entity
TrackedObject {
    id: 42,
    class: "person",
    position: Point { x: 152, y: 260 },
    first_seen: 1714567200,
    last_seen: 1714567235,
}

// 3. High-Level Structured Event
VisionEvent::ObjectEnteredZone {
    object: TrackedObject { id: 42, .. },
    zone: ZoneId("driveway"),
}
```

---

## 🎯 Example Use Cases

### 1. Smart Security & Perimeter Defense
Convert passive surveillance into proactive intelligence:
- Detect people or vehicles entering restricted zones in real time.
- Filter out nuisance triggers (weather, shadows, animals) with class confidence thresholds.
- Trigger instantaneous automated webhooks and record clips only when events occur.

### 2. Searchable Video Archive
Replace tedious manual scrubbing with structured and semantic queries:
```text
"Show every car detected between 14:00 and 16:00"
"When did the delivery truck arrive and depart?"
"Where was backpack #12 last seen?"
```

### 3. Occupancy & Spatial Analytics
Extract business intelligence and operational metrics:
- Real-time zone occupancy and customer dwell times.
- Movement heatmaps and multi-zone flow trajectories.
- Entrance and exit counters.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Language** | [Rust](https://www.rust-lang.org/) (2024 Edition) |
| **Computer Vision** | [OpenCV](https://opencv.org/) (`opencv` crate) |
| **Neural Inference** | [ONNX Runtime](https://onnxruntime.ai/) (`ort` crate), `ndarray` |
| **Concurrency & Parallelism** | `rayon`, `tokio`, standard threads |
| **Diagnostics & Telemetry** | `tracing`, `tracing-subscriber` |
| **Error Handling** | `anyhow` |
| **Persistence (Planned)** | SQLite (`rusqlite`), Vector indexing |

---

## 🚀 Quick Start

### Prerequisites

Ensure you have the following installed on your system:

- **Rust toolchain** (1.85+ recommended):
  ```bash
  curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  ```
- **OpenCV 4.x development libraries** and **Clang**:
  ```bash
  # Ubuntu / Debian
  sudo apt-get update && sudo apt-get install -y libopencv-dev clang libclang-dev pkg-config
  
  # Arch Linux
  sudo pacman -S opencv clang pkg-config
  
  # macOS (Homebrew)
  brew install opencv clang pkg-config
  ```

### Installation

Clone the repository and build in release mode:

```bash
git clone https://github.com/GuilhermeFortuna/vision-engine.git
cd vision-engine
cargo build --release
```

The optimized binary will be available at `./target/release/vision-engine`.

---

## 💻 Usage

```bash
vision-engine <video> [--model <path>]
```

### Options

| Flag / Option | Description | Default |
| :--- | :--- | :--- |
| `<video>` | Path to local input video file *(required)* | — |
| `--model <path>` | Path to custom ONNX object detection model | `models/yolov8n.onnx` |
| `-h`, `--help` | Print help and usage information | — |

### Examples

```bash
# Run with default YOLOv8n ONNX model
./target/release/vision-engine samples/traffic.mp4

# Run with a custom model
./target/release/vision-engine samples/warehouse.mp4 --model models/yolov8s-custom.onnx
```

---

## 🗺️ Roadmap & Milestones

Vision Engine is developed iteratively through vertical, verifiable milestones:

- [ ] **Milestone 1 — Foundation & Object Detection**
  - CLI argument validation and diagnostic logging.
  - Video frame decoding via OpenCV.
  - YOLO ONNX object detection via ONNX Runtime (`ort`).
  - Labeled bounding box rendering and live FPS/latency metrics.
- [ ] **Milestone 2 — Object Tracking**
  - Stable cross-frame object ID assignment and lifespan management.
  - Track trajectory smoothing and disappearance handling.
- [ ] **Milestone 3 — Pipelined Architecture**
  - Decoupled parallel stages (decode, preprocess, infer, track, render).
  - Bounded lock-free queues with backpressure management.
- [ ] **Milestone 4 — Event Engine**
  - Spatial zones and tripwire line-crossing triggers.
  - Structured event dispatch (`ObjectEnteredZone`, `ObjectCrossedLine`).
- [ ] **Milestone 5 — GPU Acceleration**
  - Zero-copy GPU memory pathways and execution backends (ROCm / CUDA / DirectML).
- [ ] **Milestone 6 — Live Stream Ingestion**
  - Real-time V4L2 USB camera (`/dev/video0`) and RTSP network stream capture.
- [ ] **Milestone 7 — Structured Persistence**
  - SQLite storage layer for sources, detections, tracks, and spatial events.
- [ ] **Milestone 8 — Video Search & Query Engine**
  - Structured SQL and natural-language / semantic search across event history.
- [ ] **Milestone 9 — Advanced Vision Capabilities**
  - Semantic segmentation, pose estimation, OCR, and re-identification embeddings.
- [ ] **Milestone 10 — Multi-Camera World Model**
  - Cross-camera identity handoff, 3D coordinate mapping, and global floorplan tracking.

---

## 🔧 Development & Validation

All contributions must adhere to strict quality and testing standards.

### CI & Git Hooks

Run the full CI pipeline locally:

```bash
./ci.sh
```

Install git hooks once per clone (pre-commit runs lint; pre-push runs full CI):

```bash
./scripts/install-git-hooks.sh
```

GitHub Actions runs the same `./ci.sh` script on every push and pull request.

### Manual validation

You can also run individual checks:

```bash
# Fast compilation check
cargo check

# Enforce formatting rules
cargo fmt --check

# Run linter with warnings as errors
cargo clippy --all-targets --all-features -- -D warnings

# Execute test suite
cargo test

# Build release target
cargo build --release
```

---

## 📄 License

This project is licensed under either the [MIT License](LICENSE-MIT) or the [Apache License, Version 2.0](LICENSE-APACHE) at your option.
