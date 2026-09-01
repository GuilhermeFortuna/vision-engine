use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

const DEFAULT_MODEL_PATH: &str = "models/yolov8n.onnx";

#[derive(Debug)]
struct Config {
    video: PathBuf,
    model: PathBuf,
}

#[derive(Debug)]
enum ParseOutcome {
    Help,
    Run(Config),
}

fn main() {
    init_tracing();
    if let Err(err) = run() {
        eprintln!("{err:?}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    match parse_args(std::env::args_os().skip(1))? {
        ParseOutcome::Help => {
            print_usage();
            Ok(())
        }
        ParseOutcome::Run(config) => {
            validate_config(&config)?;
            tracing::info!(
                video = %config.video.display(),
                model = %config.model.display(),
                "startup configuration validated"
            );
            Ok(())
        }
    }
}

fn init_tracing() {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_target(false)
        .init();
}

fn print_usage() {
    println!(
        "Usage: vision-engine <video> [--model <path>]\n\
         \n\
           <video>          Path to a local video file (required)\n\
           --model <path>   Path to ONNX model (default: {DEFAULT_MODEL_PATH})\n\
           -h, --help       Show this help"
    );
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<ParseOutcome> {
    let args: Vec<OsString> = args.into_iter().collect();

    if args.is_empty() {
        bail!("missing required argument: <video>");
    }

    for arg in &args {
        if arg == "-h" || arg == "--help" {
            return Ok(ParseOutcome::Help);
        }
    }

    let mut video: Option<PathBuf> = None;
    let mut model: Option<PathBuf> = None;
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        if arg == "--model" {
            let value = iter.next().context("missing value for option: --model")?;
            model = Some(PathBuf::from(value));
            continue;
        }

        if arg.to_string_lossy().starts_with('-') {
            bail!("unknown option: {}", arg.to_string_lossy());
        }

        if video.is_some() {
            bail!("unexpected extra argument: {}", arg.to_string_lossy());
        }

        video = Some(PathBuf::from(arg));
    }

    let video = video.context("missing required argument: <video>")?;
    let model = model.unwrap_or_else(|| PathBuf::from(DEFAULT_MODEL_PATH));

    Ok(ParseOutcome::Run(Config { video, model }))
}

fn validate_config(config: &Config) -> Result<()> {
    validate_regular_file("video", &config.video)?;
    validate_regular_file("model", &config.model)?;
    Ok(())
}

fn validate_regular_file(role: &str, path: &Path) -> Result<()> {
    let metadata = path
        .metadata()
        .with_context(|| format!("failed to read {role} path metadata"))?;

    if !metadata.is_file() {
        bail!(
            "{role} path does not exist or is not a regular file: {}",
            path.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
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

    fn os_args(args: &[&str]) -> Vec<OsString> {
        args.iter().map(OsString::from).collect()
    }

    #[test]
    fn help_long_returns_help() {
        let outcome = parse_args(os_args(&["--help"])).expect("parse should succeed");
        assert!(matches!(outcome, ParseOutcome::Help));
    }

    #[test]
    fn help_short_returns_help() {
        let outcome = parse_args(os_args(&["-h"])).expect("parse should succeed");
        assert!(matches!(outcome, ParseOutcome::Help));
    }

    #[test]
    fn video_only_uses_default_model() {
        let outcome = parse_args(os_args(&["clip.mp4"])).expect("parse should succeed");
        let ParseOutcome::Run(config) = outcome else {
            panic!("expected Run outcome");
        };
        assert_eq!(config.video, PathBuf::from("clip.mp4"));
        assert_eq!(config.model, PathBuf::from(DEFAULT_MODEL_PATH));
    }

    #[test]
    fn video_and_model_are_parsed() {
        let outcome =
            parse_args(os_args(&["clip.mp4", "--model", "m.onnx"])).expect("parse should succeed");
        let ParseOutcome::Run(config) = outcome else {
            panic!("expected Run outcome");
        };
        assert_eq!(config.video, PathBuf::from("clip.mp4"));
        assert_eq!(config.model, PathBuf::from("m.onnx"));
    }

    #[test]
    fn missing_video_is_an_error() {
        let err = parse_args(os_args(&[])).expect_err("parse should fail");
        assert!(err.to_string().contains("<video>"));
    }

    #[test]
    fn unknown_option_is_an_error() {
        let err = parse_args(os_args(&["clip.mp4", "--foo"])).expect_err("parse should fail");
        assert!(err.to_string().contains("unknown option"));
    }

    #[test]
    fn missing_model_value_is_an_error() {
        let err = parse_args(os_args(&["clip.mp4", "--model"])).expect_err("parse should fail");
        assert!(err.to_string().contains("--model"));
    }

    #[test]
    fn extra_positional_is_an_error() {
        let err = parse_args(os_args(&["a.mp4", "b.mp4"])).expect_err("parse should fail");
        assert!(err.to_string().contains("unexpected extra argument"));
    }

    #[test]
    fn valid_files_pass_validation() {
        let dir = unique_temp_dir("valid-files");
        let video = dir.join("video.mp4");
        let model = dir.join("model.onnx");
        fs::write(&video, b"video").expect("failed to write video file");
        fs::write(&model, b"model").expect("failed to write model file");

        let config = Config {
            video: video.clone(),
            model: model.clone(),
        };

        validate_config(&config).expect("validation should succeed");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_video_fails_validation() {
        let dir = unique_temp_dir("missing-video");
        let model = dir.join("model.onnx");
        fs::write(&model, b"model").expect("failed to write model file");

        let config = Config {
            video: dir.join("missing.mp4"),
            model,
        };

        let err = validate_config(&config).expect_err("validation should fail");
        assert!(err.to_string().contains("video"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn directory_as_video_fails_validation() {
        let dir = unique_temp_dir("dir-as-video");
        let model = dir.join("model.onnx");
        let video_dir = dir.join("video-dir");
        fs::create_dir(&video_dir).expect("failed to create video dir");
        fs::write(&model, b"model").expect("failed to write model file");

        let config = Config {
            video: video_dir,
            model,
        };

        let err = validate_config(&config).expect_err("validation should fail");
        assert!(err.to_string().contains("video"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_model_fails_validation() {
        let dir = unique_temp_dir("missing-model");
        let video = dir.join("video.mp4");
        fs::write(&video, b"video").expect("failed to write video file");

        let config = Config {
            video,
            model: dir.join("missing.onnx"),
        };

        let err = validate_config(&config).expect_err("validation should fail");
        assert!(err.to_string().contains("model"));
        let _ = fs::remove_dir_all(&dir);
    }
}
