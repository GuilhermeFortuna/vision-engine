# Sample media

The MP4 assets are gitignored because binary test media is kept local. The
recommended set below comes from the [Intel IoT DevKit sample-videos
repository](https://github.com/intel-iot-devkit/sample-videos), an archived
inference-sample collection released under [CC BY
4.0](https://github.com/intel-iot-devkit/sample-videos/blob/master/LICENSE).

## Recommended clips

| File | Duration | Video | Useful coverage |
| --- | ---: | --- | --- |
| `test.mp4` | 54 s | 768×432, 12 fps | People, bicycles, and cars crossing; baseline detection and tracking |
| `car-detection.mp4` | 30 s | 768×432, 12.5 fps | Vehicle detections and short, repeatable runs |
| `people-detection.mp4` | 50 s | 768×432, 12 fps | Multiple people, entrances/exits, and track association |
| `one-by-one-person-detection.mp4` | 2 m 19 s | 768×432, 10 fps | Sequential arrivals and departures; track lifecycle behavior |
| `store-aisle-detection.mp4` | 1 m 5 s | 720×404, 59.94 fps | Indoor movement, occlusion, and higher-rate decoding |
| `worker-zone-detection.mp4` | 1 m 16 s | 1920×1080, 59.94 fps | Full-HD input, zone-like movement, and throughput testing |

Download the complete recommended set with:

```bash
curl -fsSL -o samples/test.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/person-bicycle-car-detection.mp4
curl -fsSL -o samples/car-detection.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/car-detection.mp4
curl -fsSL -o samples/people-detection.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4
curl -fsSL -o samples/one-by-one-person-detection.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/one-by-one-person-detection.mp4
curl -fsSL -o samples/store-aisle-detection.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/store-aisle-detection.mp4
curl -fsSL -o samples/worker-zone-detection.mp4 \
  https://github.com/intel-iot-devkit/sample-videos/raw/master/worker-zone-detection.mp4
```

The source repository also contains specialized face, gesture, retail-object,
and driver-action clips; those are intentionally not part of the default set
because the current engine focuses on general object detection and tracking.

Run any clip with:

```bash
cargo build --release
./target/release/vision-engine samples/people-detection.mp4 --model models/yolov8n.onnx
```
