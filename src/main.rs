mod detector;

use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use detector::YoloV8Detector;
use opencv::{
    core::{Point, Scalar},
    highgui, imgproc,
    prelude::*,
    videoio::{self, VideoCapture},
};

const DEFAULT_MODEL_PATH: &str = "models/yolov8n.onnx";
const WINDOW_NAME: &str = "vision-engine";
const MIN_FPS_WINDOW: Duration = Duration::from_secs(1);

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

struct RollingFps {
    frames_in_window: u32,
    elapsed: Duration,
    latest_fps: Option<f64>,
}

impl RollingFps {
    fn new() -> Self {
        Self {
            frames_in_window: 0,
            elapsed: Duration::ZERO,
            latest_fps: None,
        }
    }

    fn record_frame(&mut self, delta: Duration) -> Option<f64> {
        self.frames_in_window += 1;
        self.elapsed += delta;

        if self.elapsed >= MIN_FPS_WINDOW {
            let fps = self.frames_in_window as f64 / self.elapsed.as_secs_f64();
            self.latest_fps = Some(fps);
            self.frames_in_window = 0;
            self.elapsed = Duration::ZERO;
            return Some(fps);
        }

        None
    }

    fn displayed_fps(&self) -> Option<f64> {
        self.latest_fps
    }
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
            run_playback(&config)
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

fn video_path_for_opencv(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("video path is not valid UTF-8: {}", path.display()))
}

fn open_video_capture(path: &Path) -> Result<VideoCapture> {
    let path = video_path_for_opencv(path)?;
    let capture = VideoCapture::from_file(path, videoio::CAP_ANY)
        .with_context(|| format!("failed to create video capture for {path}"))?;

    if !capture.is_opened()? {
        bail!("failed to open video: {path}");
    }

    Ok(capture)
}

fn should_exit(key: i32) -> bool {
    if key == -1 {
        return false;
    }

    let key = key & 0xFF;
    key == 27 || key == i32::from(b'q') || key == i32::from(b'Q')
}

fn draw_metrics_overlay(
    frame: &mut Mat,
    decode_ms: f64,
    inference_ms: f64,
    fps: Option<f64>,
) -> Result<()> {
    let decode_text = format!("Decode: {decode_ms:.1} ms");
    imgproc::put_text(
        frame,
        &decode_text,
        Point::new(10, 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.8,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    let inference_text = format!("Inference: {inference_ms:.1} ms");
    imgproc::put_text(
        frame,
        &inference_text,
        Point::new(10, 60),
        imgproc::FONT_HERSHEY_SIMPLEX,
        0.8,
        Scalar::new(0.0, 255.0, 0.0, 0.0),
        2,
        imgproc::LINE_8,
        false,
    )?;

    if let Some(fps) = fps {
        let fps_text = format!("FPS: {fps:.1}");
        imgproc::put_text(
            frame,
            &fps_text,
            Point::new(10, 90),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.8,
            Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_8,
            false,
        )?;
    }

    Ok(())
}

fn destroy_playback_window() -> Result<()> {
    highgui::destroy_window(WINDOW_NAME).context("failed to destroy display window")
}

fn run_playback(config: &Config) -> Result<()> {
    let mut capture = open_video_capture(&config.video)?;
    let mut detector = YoloV8Detector::load(&config.model)?;
    highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)
        .context("failed to create display window")?;

    let playback_result = (|| -> Result<()> {
        let mut frame = Mat::default();
        let mut rolling_fps = RollingFps::new();
        let mut last_iteration_end = Instant::now();

        loop {
            let decode_start = Instant::now();
            let read_ok = capture
                .read(&mut frame)
                .context("failed to read video frame")?;
            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

            if !read_ok || frame.empty() {
                break;
            }

            let inference_result = detector.infer(&frame)?;
            let _ = inference_result.output;
            let _ = inference_result.transform;

            let now = Instant::now();
            rolling_fps.record_frame(now.duration_since(last_iteration_end));
            last_iteration_end = now;

            draw_metrics_overlay(
                &mut frame,
                decode_ms,
                inference_result.inference_ms,
                rolling_fps.displayed_fps(),
            )?;
            highgui::imshow(WINDOW_NAME, &frame).context("failed to display video frame")?;

            if should_exit(highgui::wait_key(1)?) {
                break;
            }
        }

        Ok(())
    })();

    match destroy_playback_window() {
        Ok(()) => playback_result,
        Err(cleanup_err) => playback_result.map_err(|process_err| process_err.context(cleanup_err)),
    }
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

    #[test]
    fn fps_before_one_second() {
        let mut rolling_fps = RollingFps::new();
        let delta = Duration::from_millis(50);

        for _ in 0..10 {
            assert!(rolling_fps.record_frame(delta).is_none());
        }

        assert!(rolling_fps.displayed_fps().is_none());
    }

    #[test]
    fn fps_after_one_second() {
        let mut rolling_fps = RollingFps::new();
        let delta = Duration::from_millis(50);
        let mut last_fps = None;

        for _ in 0..20 {
            last_fps = rolling_fps.record_frame(delta);
        }

        let fps = last_fps.expect("fps should be available after one second");
        assert!((fps - 20.0).abs() < 0.1);
        assert!((rolling_fps.displayed_fps().expect("displayed fps") - 20.0).abs() < 0.1);
    }

    #[test]
    fn fps_rolling_window_resets() {
        let mut rolling_fps = RollingFps::new();
        let delta = Duration::from_millis(50);

        for _ in 0..20 {
            rolling_fps.record_frame(delta);
        }

        let first_fps = rolling_fps
            .displayed_fps()
            .expect("first window should produce fps");
        assert!((first_fps - 20.0).abs() < 0.1);

        for _ in 0..10 {
            assert!(rolling_fps.record_frame(delta).is_none());
        }

        for _ in 0..10 {
            rolling_fps.record_frame(delta);
        }

        let second_fps = rolling_fps
            .displayed_fps()
            .expect("second window should produce fps");
        assert!((second_fps - 20.0).abs() < 0.1);
        assert!((second_fps - first_fps).abs() < 0.1);
    }

    #[test]
    fn escape_exits() {
        assert!(should_exit(27));
    }

    #[test]
    fn lowercase_q_exits() {
        assert!(should_exit(113));
    }

    #[test]
    fn uppercase_q_exits() {
        assert!(should_exit(81));
    }

    #[test]
    fn no_key_continues() {
        assert!(!should_exit(-1));
    }

    #[test]
    fn other_key_continues() {
        assert!(!should_exit(65));
    }
}
