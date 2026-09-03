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

fn assert_no_backtrace(stderr: &str) {
    assert!(
        !stderr.contains("stack backtrace"),
        "expected no backtrace in stderr unless RUST_BACKTRACE is set: {stderr}"
    );
}

fn run_command(args: &[&str]) -> std::process::Output {
    let mut command = Command::new(bin());
    command.env_remove("RUST_BACKTRACE");
    for arg in args {
        command.arg(arg);
    }
    command.output().expect("failed to run vision-engine")
}

#[test]
fn help_exits_successfully() {
    let output = run_command(&["--help"]);

    assert!(output.status.success());
}

#[test]
fn missing_video_exits_with_error() {
    let output = run_command(&[]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("<video>"));
    assert_no_backtrace(&stderr);
}

#[test]
fn missing_video_file_exits_with_error() {
    let dir = unique_temp_dir("cli-missing-video");
    let model = dir.join("model.onnx");
    fs::write(&model, b"model stub").expect("failed to write model stub");

    let missing_video = dir.join("missing.mp4");
    let output = run_command(&[
        &missing_video.to_string_lossy(),
        "--model",
        &model.to_string_lossy(),
    ]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("video"),
        "expected the failing role, got: {stderr}"
    );
    assert!(
        stderr.contains(&*missing_video.to_string_lossy()),
        "expected the offending path in the error, got: {stderr}"
    );
    assert_no_backtrace(&stderr);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn directory_as_video_exits_with_error() {
    let dir = unique_temp_dir("cli-directory-video");
    let model = dir.join("model.onnx");
    fs::write(&model, b"model stub").expect("failed to write model stub");

    let output = run_command(&[&dir.to_string_lossy(), "--model", &model.to_string_lossy()]);
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("video"),
        "expected video role in error, got: {stderr}"
    );
    assert_no_backtrace(&stderr);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn invalid_onnx_exits_with_error() {
    let dir = unique_temp_dir("cli-invalid-onnx");
    let model = dir.join("invalid.onnx");
    fs::write(&model, b"not an onnx file").expect("failed to write invalid model");

    // The model is loaded before the video capture is opened, so a stub video is
    // enough to reach the ONNX failure without depending on an untracked asset.
    let video = dir.join("stub.mp4");
    fs::write(&video, b"stub video").expect("failed to write stub video");

    let output = run_command(&[
        &video.to_string_lossy(),
        "--model",
        &model.to_string_lossy(),
    ]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("failed to load ONNX model"),
        "expected ONNX load error, got: {stderr}"
    );
    assert_no_backtrace(&stderr);

    let _ = fs::remove_dir_all(&dir);
}

#[test]
fn natural_end_of_input_exits_successfully_with_max_frames() {
    let Some(video) = sample_video() else {
        eprintln!("skipping: samples/test.mp4 not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let output = run_command(&[
        &video.to_string_lossy(),
        "--model",
        &model.to_string_lossy(),
        "--max-frames",
        "5",
    ]);
    assert!(
        output.status.success(),
        "natural end should exit successfully: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

// Requires a real YOLOv8n ONNX model, so it is opt-in rather than silently skipped:
// run with `cargo test -- --ignored` and VISION_ENGINE_TEST_MODEL set.
#[test]
#[ignore = "requires VISION_ENGINE_TEST_MODEL pointing at a valid YOLOv8n ONNX model"]
fn undecodable_video_exits_with_error_when_model_is_valid() {
    let model_path = std::env::var("VISION_ENGINE_TEST_MODEL").unwrap_or_else(|_| {
        panic!("VISION_ENGINE_TEST_MODEL is not set; it must point to a valid YOLOv8n ONNX model")
    });

    if !PathBuf::from(&model_path).is_file() {
        panic!("VISION_ENGINE_TEST_MODEL does not point to a regular file: {model_path}");
    }

    let dir = unique_temp_dir("cli-undecodable-video");
    let video = dir.join("undecodable.mp4");
    fs::write(&video, b"not a video file").expect("failed to write undecodable video");

    let output = run_command(&[&video.to_string_lossy(), "--model", &model_path]);

    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("could not be decoded") || stderr.contains("failed to open video"),
        "expected decode/open error, got: {stderr}"
    );
    assert_no_backtrace(&stderr);

    let _ = fs::remove_dir_all(&dir);
}
