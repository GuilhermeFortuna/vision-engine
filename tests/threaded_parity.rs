use std::fs;
use std::path::PathBuf;
use std::process::Command;

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
fn threaded_run_matches_serial_baseline() {
    let Some(video) = sample_video() else {
        eprintln!("skipping: samples/test.mp4 not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let dump_path =
        std::env::temp_dir().join(format!("vision-engine-ve014-parity-{}", std::process::id()));
    let output = Command::new(bin())
        .arg(&video)
        .arg("--model")
        .arg(&model)
        .arg("--track-dump")
        .arg(&dump_path)
        .output()
        .expect("failed to run vision-engine");
    assert!(
        output.status.success(),
        "threaded run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let dump = fs::read_to_string(&dump_path).expect("failed to read track dump");
    let baseline =
        fs::read_to_string(repo_root().join("docs/development/baselines/VE-012/single-pass.csv"))
            .expect("failed to read VE-012 baseline");
    assert_eq!(
        dump, baseline,
        "threaded output diverged from the serial baseline"
    );

    let _ = fs::remove_file(&dump_path);
}
