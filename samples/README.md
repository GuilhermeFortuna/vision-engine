# Sample media

`test.mp4` is gitignored because binary assets are kept local. Download the
recommended acceptance clip with:

```bash
curl -fsSL -o samples/test.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4
```

**Source:** [Intel IoT DevKit sample-videos](https://github.com/intel-iot-devkit/sample-videos) — `person-bicycle-car-detection.mp4`

The clip is ~54 seconds at 768×432 and contains people, bicycles, and cars moving
through a crosswalk — well suited for COCO detection and SORT tracking demos.

**Run:**

```bash
cargo build --release
./target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx
```
