use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::Duration;

use anyhow::{Context, Result, bail};

pub const DEFAULT_MODEL_PATH: &str = "models/yolov8n.onnx";

#[derive(Debug)]
pub struct Config {
    pub video: PathBuf,
    pub model: PathBuf,
    pub loop_for: Option<Duration>,
    pub track_dump: Option<PathBuf>,
    pub max_frames: Option<u64>,
}

#[derive(Debug)]
pub enum ParseOutcome {
    Help,
    Run(Config),
}

pub fn print_usage() {
    println!(
        "Usage: vision-engine <video> [--model <path>]\n\
         \n\
         \x20 <video>          Path to a local video file (required)\n\
         \x20 --model <path>   Path to ONNX model (default: {DEFAULT_MODEL_PATH})\n\
         \x20 --loop-for-seconds <n>  Replay input until n seconds have elapsed\n\
         \x20 --track-dump <path>    Write per-frame track records to a file\n\
         \x20 --max-frames <n>       Stop after processing n frames\n\
         \x20 -h, --help       Show this help"
    );
}

pub fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<ParseOutcome> {
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
    let mut loop_for = None;
    let mut track_dump = None;
    let mut max_frames = None;
    let mut iter = args.iter();

    while let Some(arg) = iter.next() {
        if arg == "--model" {
            let value = iter.next().context("missing value for option: --model")?;
            model = Some(PathBuf::from(value));
            continue;
        }

        if arg == "--loop-for-seconds" {
            let value = iter
                .next()
                .context("missing value for option: --loop-for-seconds")?;
            let seconds = value.to_string_lossy().parse::<u64>().context(
                "invalid value for option: --loop-for-seconds (expected positive integer)",
            )?;
            if seconds == 0 {
                bail!("invalid value for option: --loop-for-seconds (expected positive integer)");
            }
            loop_for = Some(Duration::from_secs(seconds));
            continue;
        }

        if arg == "--track-dump" {
            let value = iter
                .next()
                .context("missing value for option: --track-dump")?;
            track_dump = Some(PathBuf::from(value));
            continue;
        }

        if arg == "--max-frames" {
            let value = iter
                .next()
                .context("missing value for option: --max-frames")?;
            let frames = value
                .to_string_lossy()
                .parse::<u64>()
                .context("invalid value for option: --max-frames (expected positive integer)")?;
            if frames == 0 {
                bail!("invalid value for option: --max-frames (expected positive integer)");
            }
            max_frames = Some(frames);
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

    Ok(ParseOutcome::Run(Config {
        video,
        model,
        loop_for,
        track_dump,
        max_frames,
    }))
}

pub fn validate_config(config: &Config) -> Result<()> {
    validate_regular_file("video", &config.video)?;
    validate_regular_file("model", &config.model)?;
    Ok(())
}

fn validate_regular_file(role: &str, path: &Path) -> Result<()> {
    let metadata = path
        .metadata()
        .with_context(|| format!("failed to read {role} path metadata: {}", path.display()))?;

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
    fn loop_duration_is_parsed() {
        let outcome = parse_args(os_args(&["clip.mp4", "--loop-for-seconds", "720"]))
            .expect("parse should succeed");
        let ParseOutcome::Run(config) = outcome else {
            panic!("expected Run outcome");
        };
        assert_eq!(config.loop_for, Some(Duration::from_secs(720)));
    }

    #[test]
    fn zero_loop_duration_is_an_error() {
        let err = parse_args(os_args(&["clip.mp4", "--loop-for-seconds", "0"]));
        assert!(err.is_err());
    }

    #[test]
    fn track_dump_is_parsed() {
        let outcome = parse_args(os_args(&["clip.mp4", "--track-dump", "/tmp/out.csv"]))
            .expect("parse should succeed");
        let ParseOutcome::Run(config) = outcome else {
            panic!("expected Run outcome");
        };
        assert_eq!(config.track_dump, Some(PathBuf::from("/tmp/out.csv")));
    }

    #[test]
    fn max_frames_is_parsed() {
        let outcome = parse_args(os_args(&["clip.mp4", "--max-frames", "100"]))
            .expect("parse should succeed");
        let ParseOutcome::Run(config) = outcome else {
            panic!("expected Run outcome");
        };
        assert_eq!(config.max_frames, Some(100));
    }

    #[test]
    fn zero_max_frames_is_an_error() {
        let err = parse_args(os_args(&["clip.mp4", "--max-frames", "0"]));
        assert!(err.is_err());
    }

    #[test]
    fn unparseable_max_frames_is_an_error() {
        let err = parse_args(os_args(&["clip.mp4", "--max-frames", "abc"]));
        assert!(err.is_err());
    }

    #[test]
    fn missing_track_dump_value_is_an_error() {
        let err = parse_args(os_args(&["clip.mp4", "--track-dump"]));
        assert!(err.is_err());
    }

    #[test]
    fn missing_max_frames_value_is_an_error() {
        let err = parse_args(os_args(&["clip.mp4", "--max-frames"]));
        assert!(err.is_err());
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
            loop_for: None,
            track_dump: None,
            max_frames: None,
        };

        validate_config(&config).expect("validation should succeed");
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_video_fails_validation() {
        let dir = unique_temp_dir("missing-video");
        let model = dir.join("model.onnx");
        fs::write(&model, b"model").expect("failed to write model file");

        let missing_video = dir.join("missing.mp4");
        let config = Config {
            video: missing_video.clone(),
            model,
            loop_for: None,
            track_dump: None,
            max_frames: None,
        };

        let err = validate_config(&config).expect_err("validation should fail");
        let message = err.to_string();
        assert!(
            message.contains("video"),
            "expected the failing role: {message}"
        );
        assert!(
            message.contains(&missing_video.display().to_string()),
            "expected the offending path: {message}"
        );
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
            loop_for: None,
            track_dump: None,
            max_frames: None,
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

        let missing_model = dir.join("missing.onnx");
        let config = Config {
            video,
            model: missing_model.clone(),
            loop_for: None,
            track_dump: None,
            max_frames: None,
        };

        let err = validate_config(&config).expect_err("validation should fail");
        let message = err.to_string();
        assert!(
            message.contains("model"),
            "expected the failing role: {message}"
        );
        assert!(
            message.contains(&missing_model.display().to_string()),
            "expected the offending path: {message}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
