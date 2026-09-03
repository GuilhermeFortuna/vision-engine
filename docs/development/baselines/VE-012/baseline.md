# VE-012 serial baseline

Recorded on the machine that produced commit `ea3f7588671b4ca792bbe5526e6180b512554a0e`.

## Environment

| Field | Value |
|-------|-------|
| CPU | AMD Ryzen 9 5900XT 16-Core Processor |
| Core count | 32 |
| Sample video | `samples/test.mp4` (6,031,199 bytes) |
| Model | `models/yolov8n.onnx` (12,851,107 bytes) |

## Track dumps

### Single pass (647 frames)

```bash
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --track-dump docs/development/baselines/VE-012/single-pass.csv
```

- Output: `single-pass.csv` (39,629 bytes, 647 frames processed)

### Looped run crossing rewind boundary (970 frames)

```bash
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --loop-for-seconds 3600 --max-frames 970 \
  --track-dump docs/development/baselines/VE-012/looped.csv
```

- Output: `looped.csv` (54,421 bytes, 970 frames processed, crosses one loop at frame 647)

Both dumps were regenerated after commit and verified byte-identical to the committed files.

## Throughput

Five sustained looped runs at `--loop-for-seconds 60`. FPS computed as `frames / 60` from the `playback complete` summary line.

| Run | Frames | FPS |
|-----|--------|-----|
| 1 | 1458 | 24.30 |
| 2 | 1468 | 24.46 |
| 3 | 1431 | 23.85 |
| 4 | 1397 | 23.28 |
| 5 | 1635 | 27.25 |

**Median FPS: 24.30**

```bash
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --loop-for-seconds 60
```

## Notes

- `preprocess_ms` is newly separated from `inference_ms` in the pipeline stage split; no other reported metric changed.
- This baseline is reference data for VE-014 and VE-016 parity checks, not a test fixture with pass/fail thresholds.
