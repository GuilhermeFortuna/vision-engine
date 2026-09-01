use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

const LOOPED_MAX_FRAMES: &str = "970";
const LOOPED_SECONDS: &str = "3600";
const REPEATED_RUNS: usize = 5;

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

fn read_baseline(name: &str) -> String {
    fs::read_to_string(repo_root().join(format!("docs/development/baselines/VE-012/{name}")))
        .unwrap_or_else(|err| panic!("failed to read VE-012 baseline {name}: {err}"))
}

fn run_track_dump(video: &Path, model: &Path, dump_path: &Path, extra_args: &[&str]) {
    let mut command = Command::new(bin());
    command
        .arg(video)
        .arg("--model")
        .arg(model)
        .arg("--track-dump")
        .arg(dump_path);
    for arg in extra_args {
        command.arg(arg);
    }

    let output = command.output().expect("failed to run vision-engine");
    assert!(
        output.status.success(),
        "threaded run failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn assert_byte_identical(actual: &str, expected: &str, context: &str) {
    if actual == expected {
        return;
    }

    let first_diff = actual
        .lines()
        .zip(expected.lines())
        .position(|(left, right)| left != right)
        .map(|line| line + 1);
    panic!(
        "{context}: dump diverged from baseline at line {:?}",
        first_diff
    );
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
        std::env::temp_dir().join(format!("vision-engine-ve016-single-{}", std::process::id()));
    run_track_dump(&video, &model, &dump_path, &[]);

    let dump = fs::read_to_string(&dump_path).expect("failed to read track dump");
    let baseline = read_baseline("single-pass.csv");
    assert_byte_identical(&dump, &baseline, "single-pass parity");

    let _ = fs::remove_file(&dump_path);
}

#[test]
fn looped_run_matches_serial_baseline() {
    let Some(video) = sample_video() else {
        eprintln!("skipping: samples/test.mp4 not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let dump_path =
        std::env::temp_dir().join(format!("vision-engine-ve016-looped-{}", std::process::id()));
    run_track_dump(
        &video,
        &model,
        &dump_path,
        &[
            "--loop-for-seconds",
            LOOPED_SECONDS,
            "--max-frames",
            LOOPED_MAX_FRAMES,
        ],
    );

    let dump = fs::read_to_string(&dump_path).expect("failed to read track dump");
    let baseline = read_baseline("looped.csv");
    assert_byte_identical(&dump, &baseline, "looped parity");

    let _ = fs::remove_file(&dump_path);
}

#[test]
fn repeated_threaded_runs_are_byte_identical() {
    let Some(video) = sample_video() else {
        eprintln!("skipping: samples/test.mp4 not present");
        return;
    };
    let Some(model) = sample_model() else {
        eprintln!("skipping: models/yolov8n.onnx not present");
        return;
    };

    let baseline = read_baseline("single-pass.csv");
    let mut dumps = Vec::with_capacity(REPEATED_RUNS);

    for run in 0..REPEATED_RUNS {
        let dump_path = std::env::temp_dir().join(format!(
            "vision-engine-ve016-repeat-{run}-{}",
            std::process::id()
        ));
        run_track_dump(&video, &model, &dump_path, &[]);
        let dump = fs::read_to_string(&dump_path).expect("failed to read track dump");
        let _ = fs::remove_file(&dump_path);
        dumps.push(dump);
    }

    for (index, dump) in dumps.iter().enumerate() {
        assert_byte_identical(
            dump,
            &baseline,
            &format!("repeated run {index} vs baseline"),
        );
    }

    for index in 1..dumps.len() {
        assert_byte_identical(
            &dumps[index],
            &dumps[0],
            &format!("repeated run {index} vs run 0"),
        );
    }
}
