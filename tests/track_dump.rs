use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

fn unique_temp_dir(prefix: &str) -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("vision-engine-{prefix}-{nanos}"));
    fs::create_dir_all(&dir).expect("failed to create temp dir");
    dir
}

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_vision-engine"))
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn sample_video() -> Option<PathBuf> {
    let path = repo_root().join("samples/test.mp4");
    path.is_file().then_some(path)
}

fn sample_model() -> Option<PathBuf> {
    let path = repo_root().join("models/yolov8n.onnx");
    path.is_file().then_some(path)
}

#[test]
fn track_dump_is_byte_identical_across_two_runs() {
    let Some(video) = sample_video() else {
        eprintln!("skipping: samples/test.mp4 not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let dir = unique_temp_dir("track-dump-identity");
    let dump_a = dir.join("a.csv");
    let dump_b = dir.join("b.csv");

    for dump in [&dump_a, &dump_b] {
        let output = Command::new(bin())
            .arg(&video)
            .arg("--model")
            .arg(&model)
            .arg("--max-frames")
            .arg("30")
            .arg("--track-dump")
            .arg(dump)
            .output()
            .expect("failed to run vision-engine");
        assert!(
            output.status.success(),
            "run failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    let a = fs::read(&dump_a).expect("failed to read dump a");
    let b = fs::read(&dump_b).expect("failed to read dump b");
    assert_eq!(a, b, "track dumps must be byte-identical");

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn max_frames_bounds_processed_frame_count() {
    let Some(video) = sample_video() else {
        eprintln!("skipping: samples/test.mp4 not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let dir = unique_temp_dir("max-frames-bound");
    let dump = dir.join("bounded.csv");

    let output = Command::new(bin())
        .arg(&video)
        .arg("--model")
        .arg(&model)
        .arg("--max-frames")
        .arg("5")
        .arg("--track-dump")
        .arg(&dump)
        .output()
        .expect("failed to run vision-engine");
    assert!(
        output.status.success(),
        "run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        combined.contains("frames=5") && combined.contains("playback complete"),
        "expected playback summary with frames=5, got: {combined}"
    );

    let contents = fs::read_to_string(&dump).expect("failed to read dump");
    assert!(
        contents.starts_with("frame_index,media_ms,"),
        "dump should include header"
    );

    let _ = fs::remove_dir_all(&dir);
}
