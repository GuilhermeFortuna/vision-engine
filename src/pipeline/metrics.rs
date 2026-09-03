use std::time::Duration;

use super::message::StageTimings;

const MIN_FPS_WINDOW: Duration = Duration::from_secs(1);

const STAGE_DECODE: &str = "decode";
const STAGE_PREPROCESS: &str = "preprocess";
const STAGE_INFER: &str = "infer";
const STAGE_TRACK: &str = "track";
const STAGE_RENDER: &str = "render";

const QUEUE_DECODED: &str = "decoded";
const QUEUE_PREPARED: &str = "prepared";
const QUEUE_DETECTED: &str = "detected";
const QUEUE_TRACKED: &str = "tracked";

#[derive(Debug, Clone, Copy)]
pub struct QueueDepths {
    pub decoded: (usize, usize),
    pub prepared: (usize, usize),
    pub detected: (usize, usize),
    pub tracked: (usize, usize),
}

#[derive(Debug, Clone, Copy)]
pub struct FrameMetrics {
    pub timings: StageTimings,
    pub render_ms: f64,
    pub queue_depths: QueueDepths,
    pub fps: Option<f64>,
    pub confirmed_tracks: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct StageStat {
    pub name: &'static str,
    pub mean_ms: f64,
    pub p95_ms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct QueueStat {
    pub name: &'static str,
    pub mean_depth: f64,
    pub fraction_at_capacity: f64,
}

#[derive(Debug, Clone)]
pub struct RunSummary {
    pub stages: [StageStat; 5],
    pub queues: [QueueStat; 4],
    pub slowest_stage: &'static str,
    pub frames: u64,
}

pub struct RunStats {
    decode_ms: Vec<f64>,
    preprocess_ms: Vec<f64>,
    inference_ms: Vec<f64>,
    tracking_ms: Vec<f64>,
    render_ms: Vec<f64>,
    decoded_depths: Vec<usize>,
    decoded_capacities: Vec<usize>,
    prepared_depths: Vec<usize>,
    prepared_capacities: Vec<usize>,
    detected_depths: Vec<usize>,
    detected_capacities: Vec<usize>,
    tracked_depths: Vec<usize>,
    tracked_capacities: Vec<usize>,
}

impl Default for RunStats {
    fn default() -> Self {
        Self::new()
    }
}

impl RunStats {
    pub fn new() -> Self {
        Self {
            decode_ms: Vec::new(),
            preprocess_ms: Vec::new(),
            inference_ms: Vec::new(),
            tracking_ms: Vec::new(),
            render_ms: Vec::new(),
            decoded_depths: Vec::new(),
            decoded_capacities: Vec::new(),
            prepared_depths: Vec::new(),
            prepared_capacities: Vec::new(),
            detected_depths: Vec::new(),
            detected_capacities: Vec::new(),
            tracked_depths: Vec::new(),
            tracked_capacities: Vec::new(),
        }
    }

    pub fn record(&mut self, metrics: &FrameMetrics) {
        self.decode_ms.push(metrics.timings.decode_ms);
        self.preprocess_ms.push(metrics.timings.preprocess_ms);
        self.inference_ms.push(metrics.timings.inference_ms);
        self.tracking_ms.push(metrics.timings.tracking_ms);
        self.render_ms.push(metrics.render_ms);

        self.decoded_depths.push(metrics.queue_depths.decoded.0);
        self.decoded_capacities.push(metrics.queue_depths.decoded.1);
        self.prepared_depths.push(metrics.queue_depths.prepared.0);
        self.prepared_capacities
            .push(metrics.queue_depths.prepared.1);
        self.detected_depths.push(metrics.queue_depths.detected.0);
        self.detected_capacities
            .push(metrics.queue_depths.detected.1);
        self.tracked_depths.push(metrics.queue_depths.tracked.0);
        self.tracked_capacities.push(metrics.queue_depths.tracked.1);
    }

    pub fn summary(&self) -> RunSummary {
        let decode = stage_stat(STAGE_DECODE, &self.decode_ms);
        let preprocess = stage_stat(STAGE_PREPROCESS, &self.preprocess_ms);
        let infer = stage_stat(STAGE_INFER, &self.inference_ms);
        let track = stage_stat(STAGE_TRACK, &self.tracking_ms);
        let render = stage_stat(STAGE_RENDER, &self.render_ms);
        let stages = [decode, preprocess, infer, track, render];

        let slowest_stage = stages
            .iter()
            .max_by(|left, right| {
                left.mean_ms
                    .partial_cmp(&right.mean_ms)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|stage| stage.name)
            .unwrap_or(STAGE_DECODE);

        let queues = [
            queue_stat(
                QUEUE_DECODED,
                &self.decoded_depths,
                &self.decoded_capacities,
            ),
            queue_stat(
                QUEUE_PREPARED,
                &self.prepared_depths,
                &self.prepared_capacities,
            ),
            queue_stat(
                QUEUE_DETECTED,
                &self.detected_depths,
                &self.detected_capacities,
            ),
            queue_stat(
                QUEUE_TRACKED,
                &self.tracked_depths,
                &self.tracked_capacities,
            ),
        ];

        RunSummary {
            stages,
            queues,
            slowest_stage,
            frames: self.decode_ms.len() as u64,
        }
    }
}

fn stage_stat(name: &'static str, samples: &[f64]) -> StageStat {
    let mean_ms = if samples.is_empty() {
        0.0
    } else {
        samples.iter().sum::<f64>() / samples.len() as f64
    };
    let mut sorted = samples.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(std::cmp::Ordering::Equal));
    StageStat {
        name,
        mean_ms,
        p95_ms: percentile(&sorted, 0.95),
    }
}

fn queue_stat(name: &'static str, depths: &[usize], capacities: &[usize]) -> QueueStat {
    let mean_depth = if depths.is_empty() {
        0.0
    } else {
        depths.iter().sum::<usize>() as f64 / depths.len() as f64
    };
    let at_capacity = depths
        .iter()
        .zip(capacities.iter())
        .filter(|(depth, capacity)| **depth == **capacity && **capacity > 0)
        .count();
    let fraction_at_capacity = if depths.is_empty() {
        0.0
    } else {
        at_capacity as f64 / depths.len() as f64
    };
    QueueStat {
        name,
        mean_depth,
        fraction_at_capacity,
    }
}

/// Nearest-rank percentile on a pre-sorted slice.
pub fn percentile(sorted_samples: &[f64], fraction: f64) -> f64 {
    if sorted_samples.is_empty() {
        return 0.0;
    }
    if sorted_samples.len() == 1 {
        return sorted_samples[0];
    }

    let rank = (fraction * sorted_samples.len() as f64).ceil() as usize;
    let index = rank.saturating_sub(1).min(sorted_samples.len() - 1);
    sorted_samples[index]
}

pub fn format_ms(value: Option<f64>) -> String {
    match value {
        Some(ms) if ms.is_finite() => format!("{ms:.1}"),
        _ => "--".to_string(),
    }
}

pub fn format_depth(depth: usize, capacity: usize) -> String {
    format!("{depth}/{capacity}")
}

pub struct RollingFps {
    frames_in_window: u32,
    elapsed: Duration,
    latest_fps: Option<f64>,
}

impl RollingFps {
    pub fn new() -> Self {
        Self {
            frames_in_window: 0,
            elapsed: Duration::ZERO,
            latest_fps: None,
        }
    }

    pub fn record_frame(&mut self, delta: Duration) -> Option<f64> {
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

    pub fn displayed_fps(&self) -> Option<f64> {
        self.latest_fps
    }
}

impl Default for RollingFps {
    fn default() -> Self {
        Self::new()
    }
}

pub fn log_instrumentation_summary(summary: &RunSummary) {
    let [decode, preprocess, infer, track, render] = summary.stages;
    let [decoded, prepared, detected, tracked] = summary.queues;

    tracing::info!(
        frames = summary.frames,
        slowest_stage = summary.slowest_stage,
        decode_mean_ms = decode.mean_ms,
        decode_p95_ms = decode.p95_ms,
        preprocess_mean_ms = preprocess.mean_ms,
        preprocess_p95_ms = preprocess.p95_ms,
        infer_mean_ms = infer.mean_ms,
        infer_p95_ms = infer.p95_ms,
        track_mean_ms = track.mean_ms,
        track_p95_ms = track.p95_ms,
        render_mean_ms = render.mean_ms,
        render_p95_ms = render.p95_ms,
        decoded_queue = decoded.name,
        decoded_mean_depth = decoded.mean_depth,
        decoded_fraction_at_capacity = decoded.fraction_at_capacity,
        prepared_queue = prepared.name,
        prepared_mean_depth = prepared.mean_depth,
        prepared_fraction_at_capacity = prepared.fraction_at_capacity,
        detected_queue = detected.name,
        detected_mean_depth = detected.mean_depth,
        detected_fraction_at_capacity = detected.fraction_at_capacity,
        tracked_queue = tracked.name,
        tracked_mean_depth = tracked.mean_depth,
        tracked_fraction_at_capacity = tracked.fraction_at_capacity,
        "instrumentation summary"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    struct SampleMetrics {
        decode_ms: f64,
        preprocess_ms: f64,
        inference_ms: f64,
        tracking_ms: f64,
        render_ms: f64,
        decoded: (usize, usize),
        prepared: (usize, usize),
        detected: (usize, usize),
        tracked: (usize, usize),
    }

    fn sample_metrics(sample: SampleMetrics) -> FrameMetrics {
        FrameMetrics {
            timings: StageTimings {
                decode_ms: sample.decode_ms,
                preprocess_ms: sample.preprocess_ms,
                inference_ms: sample.inference_ms,
                tracking_ms: sample.tracking_ms,
            },
            render_ms: sample.render_ms,
            queue_depths: QueueDepths {
                decoded: sample.decoded,
                prepared: sample.prepared,
                detected: sample.detected,
                tracked: sample.tracked,
            },
            fps: None,
            confirmed_tracks: 0,
        }
    }

    #[test]
    fn percentile_p95_of_one_through_one_hundred() {
        let samples: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((percentile(&samples, 0.95) - 95.0).abs() < f64::EPSILON);
    }

    #[test]
    fn percentile_single_sample_returns_that_sample() {
        assert!((percentile(&[42.0], 0.95) - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn percentile_empty_returns_zero() {
        assert!((percentile(&[], 0.95)).abs() < f64::EPSILON);
    }

    #[test]
    fn format_depth_empty_and_full() {
        assert_eq!(format_depth(0, 2), "0/2");
        assert_eq!(format_depth(2, 2), "2/2");
    }

    #[test]
    fn format_ms_one_decimal_and_unavailable() {
        assert_eq!(format_ms(Some(12.34)), "12.3");
        assert_eq!(format_ms(None), "--");
        assert_eq!(format_ms(Some(f64::NAN)), "--");
    }

    #[test]
    fn run_stats_summary_selects_slowest_stage_and_saturation() {
        let mut stats = RunStats::new();
        stats.record(&sample_metrics(SampleMetrics {
            decode_ms: 1.0,
            preprocess_ms: 10.0,
            inference_ms: 2.0,
            tracking_ms: 1.0,
            render_ms: 1.0,
            decoded: (0, 2),
            prepared: (0, 2),
            detected: (0, 2),
            tracked: (0, 2),
        }));
        stats.record(&sample_metrics(SampleMetrics {
            decode_ms: 1.0,
            preprocess_ms: 12.0,
            inference_ms: 2.0,
            tracking_ms: 1.0,
            render_ms: 1.0,
            decoded: (2, 2),
            prepared: (0, 2),
            detected: (0, 2),
            tracked: (0, 2),
        }));
        stats.record(&sample_metrics(SampleMetrics {
            decode_ms: 1.0,
            preprocess_ms: 11.0,
            inference_ms: 2.0,
            tracking_ms: 1.0,
            render_ms: 1.0,
            decoded: (2, 2),
            prepared: (0, 2),
            detected: (0, 2),
            tracked: (0, 2),
        }));

        let summary = stats.summary();
        assert_eq!(summary.slowest_stage, STAGE_PREPROCESS);
        assert!((summary.stages[1].mean_ms - 11.0).abs() < f64::EPSILON);
        assert!((summary.queues[0].fraction_at_capacity - (2.0 / 3.0)).abs() < f64::EPSILON);
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
}
