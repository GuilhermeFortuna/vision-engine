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

#[test]
fn help_exits_successfully() {
    let output = Command::new(bin())
        .arg("--help")
        .output()
        .expect("failed to run vision-engine");

    assert!(output.status.success());
}

#[test]
fn missing_video_exits_with_error() {
    let output = Command::new(bin())
        .output()
        .expect("failed to run vision-engine");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("<video>"));
}

#[test]
fn missing_video_file_exits_with_error() {
    let dir = unique_temp_dir("cli-missing-video");
    let model = dir.join("model.onnx");
    fs::write(&model, b"model stub").expect("failed to write model stub");

    let missing_video = dir.join("missing.mp4");
    let output = Command::new(bin())
        .arg(&missing_video)
        .arg("--model")
        .arg(&model)
        .output()
        .expect("failed to run vision-engine");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("video"));

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn invalid_onnx_exits_with_error() {
    let dir = unique_temp_dir("cli-invalid-onnx");
    let model = dir.join("invalid.onnx");
    fs::write(&model, b"not an onnx file").expect("failed to write invalid model");

    let video = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("samples/test.mp4");
    assert!(
        video.is_file(),
        "samples/test.mp4 is required for invalid_onnx_exits_with_error"
    );

    let output = Command::new(bin())
        .arg(&video)
        .arg("--model")
        .arg(&model)
        .output()
        .expect("failed to run vision-engine");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to load ONNX model"),
        "expected ONNX load error, got: {stderr}"
    );

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn undecodable_video_exits_with_error_when_model_is_valid() {
    let model_path = std::env::var("VISION_ENGINE_TEST_MODEL").ok();
    let Some(model_path) = model_path else {
        eprintln!(
            "skipped undecodable_video_exits_with_error_when_model_is_valid: set VISION_ENGINE_TEST_MODEL to a valid YOLOv8n ONNX path"
        );
        return;
    };

    if !PathBuf::from(&model_path).is_file() {
        panic!("VISION_ENGINE_TEST_MODEL does not point to a regular file: {model_path}");
    }

    let dir = unique_temp_dir("cli-undecodable-video");
    let video = dir.join("undecodable.mp4");
    fs::write(&video, b"not a video file").expect("failed to write undecodable video");

    let output = Command::new(bin())
        .arg(&video)
        .arg("--model")
        .arg(&model_path)
        .output()
        .expect("failed to run vision-engine");

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("could not be decoded") || stderr.contains("failed to open video"),
        "expected decode/open error, got: {stderr}"
    );

    let _ = fs::remove_dir_all(&dir);
}
