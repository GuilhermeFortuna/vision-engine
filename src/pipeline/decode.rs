use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use opencv::{
    prelude::*,
    videoio::{self, VideoCapture},
};
use vision_engine::tracking::{FrameClock, FrameStamp, TimeSource};

use super::message::{DecodedFrame, StageTimings};

const FALLBACK_FRAME_INTERVAL_MS: f64 = 1000.0 / 30.0;

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlaybackFrameOutcome {
    Continue,
    EndOfVideo,
    Undecodable,
}

pub struct DecodeStage {
    capture: VideoCapture,
    frame_clock: FrameClock,
    source_fps: Option<f64>,
    loop_for: Option<Duration>,
    max_frames: Option<u64>,
    run_started: Instant,
    media_offset_ms: f64,
    provenance_counts: ProvenanceCounts,
    fallback_reported: bool,
    last_stamp: Option<FrameStamp>,
    video_path: String,
}

pub enum DecodeOutcome {
    Frame { stamp: FrameStamp, decode_ms: f64 },
    EndOfRun,
}

#[derive(Debug, Clone, Copy)]
pub struct DecodeSummary {
    pub last_stamp: Option<FrameStamp>,
    pub frame_count: u64,
    pub reported: u64,
    pub derived_from_frame_rate: u64,
    pub derived_from_index: u64,
    pub adjustments: u64,
}

impl DecodeStage {
    pub fn open(video: &Path, loop_for: Option<Duration>, max_frames: Option<u64>) -> Result<Self> {
        let mut capture = open_video_capture(video)?;
        let source_fps = read_capture_fps(&mut capture)?;
        let video_path = video.display().to_string();

        Ok(Self {
            capture,
            frame_clock: FrameClock::new(source_fps),
            source_fps,
            loop_for,
            max_frames,
            run_started: Instant::now(),
            media_offset_ms: 0.0,
            provenance_counts: ProvenanceCounts::default(),
            fallback_reported: false,
            last_stamp: None,
            video_path,
        })
    }

    pub fn next(&mut self) -> Result<Option<DecodedFrame>> {
        let mut frame = Mat::default();
        match self.next_into(&mut frame)? {
            DecodeOutcome::EndOfRun => Ok(None),
            DecodeOutcome::Frame { stamp, decode_ms } => Ok(Some(DecodedFrame {
                frame,
                stamp,
                timings: StageTimings {
                    decode_ms,
                    ..StageTimings::default()
                },
            })),
        }
    }

    pub fn next_into(&mut self, frame: &mut Mat) -> Result<DecodeOutcome> {
        if self
            .loop_for
            .is_some_and(|limit| self.run_started.elapsed() >= limit)
        {
            return Ok(DecodeOutcome::EndOfRun);
        }

        if self
            .max_frames
            .is_some_and(|limit| self.frame_clock.stamped_count() >= limit)
        {
            return Ok(DecodeOutcome::EndOfRun);
        }

        loop {
            let decode_start = Instant::now();
            let read_ok = self
                .capture
                .read(frame)
                .context("failed to read video frame")?;
            let decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;

            match classify_playback_frame(
                read_ok,
                frame.empty(),
                self.frame_clock.stamped_count() as u32,
            ) {
                PlaybackFrameOutcome::Continue => {}
                PlaybackFrameOutcome::EndOfVideo => {
                    if self.loop_for.is_none() {
                        return Ok(DecodeOutcome::EndOfRun);
                    }
                    let rewound = self
                        .capture
                        .set(videoio::CAP_PROP_POS_FRAMES, 0.0)
                        .context("failed to rewind video for sustained run")?;
                    if !rewound {
                        bail!("failed to rewind video for sustained run: backend rejected seek");
                    }
                    self.media_offset_ms = self
                        .last_stamp
                        .map(|stamp| stamp.media_ms + frame_interval_ms(self.source_fps))
                        .unwrap_or(self.media_offset_ms);
                    continue;
                }
                PlaybackFrameOutcome::Undecodable => {
                    bail!("video file could not be decoded: {}", self.video_path);
                }
            }

            let reported_ms = read_capture_pos_msec(&mut self.capture)?
                .map(|position_ms| position_ms + self.media_offset_ms);
            let stamp = self.frame_clock.stamp(reported_ms);
            if stamp.source != TimeSource::Reported && !self.fallback_reported {
                self.fallback_reported = true;
                tracing::warn!(
                    frame_index = stamp.index,
                    source = ?stamp.source,
                    "tracking timestamp unavailable; using deterministic fallback for this run"
                );
            }
            self.provenance_counts.record(stamp.source);
            self.last_stamp = Some(stamp);

            return Ok(DecodeOutcome::Frame { stamp, decode_ms });
        }
    }

    pub fn summary(&self) -> DecodeSummary {
        DecodeSummary {
            last_stamp: self.last_stamp,
            frame_count: self.frame_clock.stamped_count(),
            reported: self.provenance_counts.reported,
            derived_from_frame_rate: self.provenance_counts.derived_from_frame_rate,
            derived_from_index: self.provenance_counts.derived_from_index,
            adjustments: self.frame_clock.adjustments(),
        }
    }
}

pub fn log_playback_summary(summary: &DecodeSummary, rejected_filter_updates: u64) {
    let media_ms = summary
        .last_stamp
        .map(|stamp| stamp.media_ms)
        .unwrap_or(0.0);
    tracing::info!(
        frames = summary.frame_count,
        media_ms,
        reported = summary.reported,
        derived_fps = summary.derived_from_frame_rate,
        derived_index = summary.derived_from_index,
        adjustments = summary.adjustments,
        rejected_filter_updates,
        "playback complete"
    );
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

pub(crate) fn frame_interval_ms(source_fps: Option<f64>) -> f64 {
    source_fps
        .filter(|fps| fps.is_finite() && *fps > 0.0)
        .map(|fps| 1000.0 / fps)
        .unwrap_or(FALLBACK_FRAME_INTERVAL_MS)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn unique_temp_dir(prefix: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| duration.as_nanos())
            .unwrap_or(0);
        let dir = std::env::temp_dir().join(format!("vision-engine-{prefix}-{nanos}"));
        fs::create_dir_all(&dir).expect("failed to create temp dir");
        dir
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

    #[test]
    fn open_on_missing_file_produces_existing_error() {
        let dir = unique_temp_dir("decode-missing");
        let missing = dir.join("missing.mp4");
        let err = match DecodeStage::open(&missing, None, None) {
            Ok(_) => panic!("should fail"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(
            message.contains("video"),
            "expected video role in error: {message}"
        );
        assert!(
            message.contains(&missing.display().to_string()),
            "expected path in error: {message}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_on_directory_produces_existing_error() {
        let dir = unique_temp_dir("decode-dir");
        let err = match DecodeStage::open(&dir, None, None) {
            Ok(_) => panic!("should fail"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(
            message.contains("video"),
            "expected video role in error: {message}"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_on_undecodable_file_produces_existing_error() {
        let dir = unique_temp_dir("decode-undecodable");
        let path = dir.join("bad.mp4");
        fs::write(&path, b"not a video").expect("failed to write file");
        let Ok(mut stage) = DecodeStage::open(&path, None, None) else {
            // Some backends reject invalid containers at open rather than first read.
            let _ = fs::remove_dir_all(&dir);
            return;
        };
        let err = match stage.next() {
            Ok(None) => panic!("expected error for undecodable file"),
            Ok(Some(_)) => panic!("expected error for undecodable file"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("video file could not be decoded"),
            "expected undecodable message: {err}"
        );
        let _ = fs::remove_dir_all(&dir);
    }
}
