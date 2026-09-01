use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use vision_engine::detector::Detection;
use vision_engine::tracking::{FrameStamp, Track, Tracker};

const PROGRESS_LOG_INTERVAL: Duration = Duration::from_secs(1);

pub struct TrackStage {
    tracker: Tracker,
    last_progress_log: Option<Instant>,
    run_started: Instant,
}

pub struct Tracked {
    pub tracks: Vec<Track>,
    pub tracking_ms: f64,
}

impl Default for TrackStage {
    fn default() -> Self {
        Self::new()
    }
}

impl TrackStage {
    pub fn new() -> Self {
        Self {
            tracker: Tracker::new(),
            last_progress_log: None,
            run_started: Instant::now(),
        }
    }

    pub fn update(&mut self, detections: &[Detection], stamp: FrameStamp) -> Result<Tracked> {
        let tracking_start = Instant::now();
        let tracks = self
            .tracker
            .try_update(detections, stamp)
            .with_context(|| format!("tracking update failed at frame {}", stamp.index))?;
        let tracking_ms = tracking_start.elapsed().as_secs_f64() * 1000.0;
        Ok(Tracked {
            tracks,
            tracking_ms,
        })
    }

    pub fn live_track_count(&self) -> usize {
        self.tracker.live_track_count()
    }

    pub fn rejected_updates(&self) -> u64 {
        self.tracker.rejected_updates()
    }

    pub fn maybe_log_progress(&mut self, stamp: FrameStamp, confirmed_tracks: usize) {
        let now = Instant::now();
        if self
            .last_progress_log
            .is_none_or(|last| now.duration_since(last) >= PROGRESS_LOG_INTERVAL)
        {
            tracing::info!(
                elapsed_seconds = self.run_started.elapsed().as_secs_f64(),
                frame_count = stamp.index + 1,
                live_tracks = self.live_track_count(),
                confirmed_tracks,
                "tracking progress"
            );
            self.last_progress_log = Some(now);
        }
    }
}
