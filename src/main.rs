mod render;

use std::ffi::OsString;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use opencv::{
    highgui,
    prelude::*,
    videoio::{self, VideoCapture},
};
use render::FrameMetrics;
use vision_engine::detector::YoloV8Detector;
use vision_engine::tracking::{FrameClock, FrameStamp, TimeSource, TrackState, Tracker};

const DEFAULT_MODEL_PATH: &str = "models/yolov8n.onnx";
const DEFAULT_LOG_FILTER: &str = "info,ort=warn";
const WINDOW_NAME: &str = "vision-engine";
const MIN_FPS_WINDOW: Duration = Duration::from_secs(1);
const PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(1);
const FALLBACK_FRAME_INTERVAL_MS: f64 = 1000.0 / 30.0;

#[derive(Debug)]
struct Config {
    video: PathBuf,
    model: PathBuf,
    loop_for: Option<Duration>,
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
    // ONNX Runtime bridges its own verbose INFO logging into `tracing` under the
    // `ort` target, which buries application output. Keep it at `warn` unless the
    // user asks for more through `RUST_LOG`.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(DEFAULT_LOG_FILTER));

    // Colour codes are written between the field name and `=`, which makes
    // redirected logs unparseable by ordinary text tooling. Emit them only when
    // stdout is actually a terminal.
    let use_ansi = std::io::stdout().is_terminal();

    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .with_ansi(use_ansi)
        .init();
}

fn print_usage() {
    println!(
        "Usage: vision-engine <video> [--model <path>]\n\
         \n\
         \x20 <video>          Path to a local video file (required)\n\
         \x20 --model <path>   Path to ONNX model (default: {DEFAULT_MODEL_PATH})\n\
         \x20 --loop-for-seconds <n>  Replay input until n seconds have elapsed\n\
         \x20 -h, --help       Show this help"
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
    let mut loop_for = None;
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
    }))
}

fn validate_config(config: &Config) -> Result<()> {
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

fn video_path_for_opencv(path: &Path) -> Result<&str> {
    path.to_str()
        .with_context(|| format!("video path is not valid UTF-8: {}", path.display()))
}

fn sanitize_capture_f64(value: f64) -> Option<f64> {
    if value.is_finite() && value >= 0.0 {
        Some(value)
    } else {
        None
    }
}

fn read_capture_fps(capture: &mut VideoCapture) -> Result<Option<f64>> {
    let fps = capture.get(videoio::CAP_PROP_FPS)?;
    Ok(match fps {
        value if value.is_finite() && value > 0.0 => Some(value),
        _ => None,
    })
}

fn read_capture_pos_msec(capture: &mut VideoCapture) -> Result<Option<f64>> {
    let pos_msec = capture.get(videoio::CAP_PROP_POS_MSEC)?;
    Ok(sanitize_capture_f64(pos_msec))
}

fn frame_interval_ms(source_fps: Option<f64>) -> f64 {
    source_fps
        .filter(|fps| fps.is_finite() && *fps > 0.0)
        .map(|fps| 1000.0 / fps)
        .unwrap_or(FALLBACK_FRAME_INTERVAL_MS)
}

#[derive(Debug, Default)]
struct ProvenanceCounts {
    reported: u64,
    derived_from_frame_rate: u64,
    derived_from_index: u64,
}

impl ProvenanceCounts {
    fn record(&mut self, source: TimeSource) {
        match source {
            TimeSource::Reported => self.reported += 1,
            TimeSource::DerivedFromFrameRate => self.derived_from_frame_rate += 1,
            TimeSource::DerivedFromIndex => self.derived_from_index += 1,
        }
    }
}

fn log_playback_summary(
    last_stamp: Option<FrameStamp>,
    frame_count: u64,
    provenance: &ProvenanceCounts,
    adjustments: u64,
    rejected_filter_updates: u64,
) {
    let media_ms = last_stamp.map(|stamp| stamp.media_ms).unwrap_or(0.0);
    tracing::info!(
        frames = frame_count,
        media_ms,
        reported = provenance.reported,
        derived_fps = provenance.derived_from_frame_rate,
        derived_index = provenance.derived_from_index,
        adjustments,
        rejected_filter_updates,
        "playback complete"
    );
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaybackFrameOutcome {
    Continue,
    EndOfVideo,
    Undecodable,
}

fn classify_playback_frame(
    read_ok: bool,
    frame_empty: bool,
    frames_decoded: u32,
) -> PlaybackFrameOutcome {
    if read_ok && !frame_empty {
        return PlaybackFrameOutcome::Continue;
    }

    if frames_decoded == 0 {
        PlaybackFrameOutcome::Undecodable
    } else {
        PlaybackFrameOutcome::EndOfVideo
    }
}

fn should_exit(key: i32) -> bool {
    if key == -1 {
        return false;
    }

    let key = key & 0xFF;
    key == 27 || key == i32::from(b'q') || key == i32::from(b'Q')
}

fn destroy_playback_window() -> Result<()> {
    highgui::destroy_window(WINDOW_NAME).context("failed to destroy display window")
}

fn run_playback(config: &Config) -> Result<()> {
    // Load the model before opening the video so an unsupported model is reported
    // without depending on a decodable input, which keeps startup tests asset-free.
    let mut detector = YoloV8Detector::load(&config.model)?;
    let mut capture = open_video_capture(&config.video)?;
    let source_fps = read_capture_fps(&mut capture)?;
    highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)
        .context("failed to create display window")?;

    let video_path = config.video.display().to_string();
    let playback_result = (|| -> Result<()> {
        let mut frame = Mat::default();
        let mut rolling_fps = RollingFps::new();
        let mut last_iteration_end = Instant::now();
        let mut frame_clock = FrameClock::new(source_fps);
        let mut provenance_counts = ProvenanceCounts::default();
        let mut last_stamp = None;
        let mut tracker = Tracker::new();
        let run_started = Instant::now();
        let mut last_progress_log = None;
        let mut media_offset_ms = 0.0;
        let mut fallback_reported = false;

        loop {
            if config
                .loop_for
                .is_some_and(|limit| run_started.elapsed() >= limit)
            {
                break;
            }

            let decode_start = Instant::now();
            let read_ok = capture
                .read(&mut frame)
                .context("failed to read video frame")?;
            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

            match classify_playback_frame(
                read_ok,
                frame.empty(),
                frame_clock.stamped_count() as u32,
            ) {
                PlaybackFrameOutcome::Continue => {}
                PlaybackFrameOutcome::EndOfVideo => {
                    if config.loop_for.is_none() {
                        break;
                    }
                    let rewound = capture
                        .set(videoio::CAP_PROP_POS_FRAMES, 0.0)
                        .context("failed to rewind video for sustained run")?;
                    if !rewound {
                        bail!("failed to rewind video for sustained run: backend rejected seek");
                    }
                    media_offset_ms = last_stamp
                        .map(|stamp: FrameStamp| stamp.media_ms + frame_interval_ms(source_fps))
                        .unwrap_or(media_offset_ms);
                    continue;
                }
                PlaybackFrameOutcome::Undecodable => {
                    bail!("video file could not be decoded: {video_path}");
                }
            }

            let reported_ms = read_capture_pos_msec(&mut capture)?
                .map(|position_ms| position_ms + media_offset_ms);
            let stamp = frame_clock.stamp(reported_ms);
            // One line per run, not per frame: the end-of-run summary reports the
            // full provenance breakdown, so repeating this every frame only floods
            // the console on a source that never reports timestamps.
            if stamp.source != TimeSource::Reported && !fallback_reported {
                fallback_reported = true;
                tracing::warn!(
                    frame_index = stamp.index,
                    source = ?stamp.source,
                    "tracking timestamp unavailable; using deterministic fallback for this run"
                );
            }
            provenance_counts.record(stamp.source);
            last_stamp = Some(stamp);

            let inference_result = detector.infer(&frame)?;

            let tracking_start = Instant::now();
            let tracks = tracker
                .try_update(&inference_result.detections, stamp)
                .with_context(|| format!("tracking update failed at frame {}", stamp.index))?;
            let tracking_ms = tracking_start.elapsed().as_secs_f64() * 1000.0;

            let now = Instant::now();
            rolling_fps.record_frame(now.duration_since(last_iteration_end));
            last_iteration_end = now;

            let confirmed_tracks = tracks
                .iter()
                .filter(|track| track.state == TrackState::Confirmed)
                .count();

            if last_progress_log
                .is_none_or(|last| now.duration_since(last) >= PROGRESS_LOG_INTERVAL)
            {
                tracing::info!(
                    elapsed_seconds = run_started.elapsed().as_secs_f64(),
                    frame_count = stamp.index + 1,
                    live_tracks = tracker.live_track_count(),
                    confirmed_tracks,
                    "tracking progress"
                );
                last_progress_log = Some(now);
            }

            render::draw_tracks(&mut frame, &tracks).context("failed to draw track overlays")?;
            render::draw_metrics_overlay(
                &mut frame,
                &FrameMetrics {
                    decode_ms,
                    inference_ms: inference_result.inference_ms,
                    tracking_ms,
                    fps: rolling_fps.displayed_fps(),
                    confirmed_tracks,
                },
            )
            .context("failed to draw performance metrics overlay")?;
            highgui::imshow(WINDOW_NAME, &frame).context("failed to display video frame")?;

            let key = highgui::wait_key(1).context("failed to poll keyboard events")?;
            if should_exit(key) {
                break;
            }
        }

        log_playback_summary(
            last_stamp,
            frame_clock.stamped_count(),
            &provenance_counts,
            frame_clock.adjustments(),
            tracker.rejected_updates(),
        );

        Ok(())
    })();

    let Err(cleanup_err) = destroy_playback_window() else {
        return playback_result;
    };

    match playback_result {
        // Nothing else failed, so the cleanup failure is the failure to report.
        Ok(()) => Err(cleanup_err),
        // The processing failure stays the reported error; cleanup is retained
        // alongside it rather than displacing it as the headline.
        Err(process_err) => {
            tracing::error!(error = %format!("{cleanup_err:#}"), "display window cleanup failed");
            Err(process_err)
        }
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
    fn frame_interval_uses_source_fps_when_available() {
        assert!((frame_interval_ms(Some(25.0)) - 40.0).abs() < f64::EPSILON);
    }

    #[test]
    fn frame_interval_uses_nominal_fallback_when_source_fps_is_missing() {
        assert!((frame_interval_ms(None) - FALLBACK_FRAME_INTERVAL_MS).abs() < f64::EPSILON);
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

    #[test]
    fn continue_when_frame_read_succeeds() {
        assert_eq!(
            classify_playback_frame(true, false, 0),
            PlaybackFrameOutcome::Continue
        );
        assert_eq!(
            classify_playback_frame(true, false, 10),
            PlaybackFrameOutcome::Continue
        );
    }

    #[test]
    fn end_of_video_after_at_least_one_frame() {
        assert_eq!(
            classify_playback_frame(false, false, 1),
            PlaybackFrameOutcome::EndOfVideo
        );
        assert_eq!(
            classify_playback_frame(true, true, 5),
            PlaybackFrameOutcome::EndOfVideo
        );
        assert_eq!(
            classify_playback_frame(false, true, 3),
            PlaybackFrameOutcome::EndOfVideo
        );
    }

    #[test]
    fn undecodable_when_no_frames_were_read() {
        assert_eq!(
            classify_playback_frame(false, false, 0),
            PlaybackFrameOutcome::Undecodable
        );
        assert_eq!(
            classify_playback_frame(true, true, 0),
            PlaybackFrameOutcome::Undecodable
        );
        assert_eq!(
            classify_playback_frame(false, true, 0),
            PlaybackFrameOutcome::Undecodable
        );
    }
}
