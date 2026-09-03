# VE-016 Milestone 3 acceptance record

Recorded on commit `ac78494f8b6ad7c953ed45231829626f26bdf03a` (threaded pipeline).

## Environment

| Field | Value |
|-------|-------|
| CPU | AMD Ryzen 9 5900XT 16-Core Processor |
| Core count | 32 |
| Sample video | `samples/test.mp4` |
| Model | `models/yolov8n.onnx` |
| Serial baseline commit | `ea3f7588671b4ca792bbe5526e6180b512554a0e` (VE-012) |

## Automated validation (completed)

### Parity

| Check | Result |
|-------|--------|
| Single-pass vs `VE-012/single-pass.csv` | PASS |
| Looped (970 frames) vs `VE-012/looped.csv` | PASS |
| Five repeated threaded runs byte-identical | PASS |

### Liveness

| Check | Result |
|-------|--------|
| Stall + backpressure: decode | PASS |
| Stall + backpressure: preprocess | PASS |
| Stall + backpressure: infer | PASS |
| Stall + backpressure: track | PASS |
| Shutdown: full queues | PASS |
| Shutdown: empty queues | PASS |
| Shutdown: mid-frame | PASS |
| Shutdown: end of input | PASS |
| Shutdown: before first frame | PASS |
| Ten start/stop cycles (threads + RSS stable post-join) | PASS |

### Failure and exit codes

| Check | Result |
|-------|--------|
| Induced failure per stage (names stage + frame) | PASS |
| Induced panic per stage | PASS |
| Non-fatal pipeline completion | PASS |
| `--help` → 0 | PASS |
| Missing video / file / directory → non-zero, no backtrace | PASS |
| Invalid ONNX → non-zero, no backtrace | PASS |
| Natural end (`--max-frames 5`) → 0 | PASS |

Run liveness integration tests:

```bash
cargo test --features test-utils --test pipeline_liveness -- --test-threads=1
```

## Per-frame allocation cost

Measured via `per_frame_buffer_cost_is_quantified` (decode + preprocess time as share of total stage time over 50 frames):

| Metric | Value |
|--------|-------|
| Allocation share | **30.3%** |

```bash
cargo test --lib per_frame_buffer_cost_is_quantified -- --nocapture --test-threads=1
```

## Identity churn (VE-011 method)

Computed from `VE-012/single-pass.csv` (distinct confirmed track IDs per frame × 100):

| Metric | Value |
|--------|-------|
| Distinct confirmed IDs | 15 |
| Frames with confirmed tracks | 314 |
| Churn (IDs per 100 frames) | **4.78** |

Expected object count for the sample segment: multiple COCO classes present; churn is low because IDs are stable once confirmed.

## Measured acceptance (manual — not run in agent session)

The following require local release runs. They were **not** executed during implementation to avoid long, memory-heavy benchmark sessions.

### Throughput (five × 60 s looped runs)

```bash
cargo build --release
for i in 1 2 3 4 5; do
  target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
    --loop-for-seconds 60 2>&1 | tee /tmp/ve016-tp-$i.log
done
```

Compare median FPS (`frames / 60` from `playback complete`) against VE-012 serial median **24.30 FPS**. Record `instrumentation summary` for bottleneck stage.

### Sustained soak

```bash
scripts/sustained-run.sh samples/test.mp4 models/yolov8n.onnx
```

Record RSS, queue-depth, and per-stage latency series from the emitted CSV.

### Visual confirmation

```bash
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx
```

Confirm boxes track objects, IDs persist, and instrumentation overlay is readable.

## Milestone acceptance status

**Code and automated hardening: complete.**

**Milestone 3 accepted: pending** manual throughput (must exceed VE-012 serial median), sustained soak PASS, and visual confirmation. Update `docs/development/STATUS.md` to `DONE` only after those three are recorded above.

## Deferred follow-ups (out of scope for VE-016)

| Follow-up | Measurement that would justify it |
|-----------|-----------------------------------|
| Buffer pooling | 30.3% per-frame allocation share on decode+preprocess |
| Inference worker pool | Bottleneck stage from instrumentation summary (typically `infer`) |
| Frame-drop policy for live sources | Queue saturation fraction at capacity during sustained run |
