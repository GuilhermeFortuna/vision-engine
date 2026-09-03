mod decode;
mod infer;
mod message;
mod metrics;
mod preprocess;
mod queue;
mod render;
pub mod runtime;
mod track;
mod track_dump;

use std::time::{Duration, Instant};

use crate::detector::LoadedModel;
use crate::tracking::TrackState;
use anyhow::Result;

use crate::cli::Config;

pub use queue::QUEUE_CAPACITY;
#[cfg(any(test, feature = "test-utils"))]
pub use runtime::test_support;
pub use runtime::{FaultConfig, Pipeline, PipelineRunStats, Stage};

use decode::log_playback_summary;
use metrics::{FrameMetrics, RollingFps, RunStats, format_depth, log_instrumentation_summary};
use render::{Presentation, RenderStage};
use track_dump::TrackDump;

const INSTRUMENTATION_LOG_INTERVAL: Duration = Duration::from_secs(1);

pub fn run(config: &Config) -> Result<()> {
    let model = LoadedModel::load(&config.model)?;
    let mut render = RenderStage::open()?;
    let pipeline = Pipeline::spawn(config, model)?;
    let mut rolling_fps = RollingFps::new();
    let mut last_iteration_end = Instant::now();
    let mut run_stats = RunStats::new();
    let mut last_instrumentation_log = None;
    let mut last_render_ms: Option<f64> = None;
    let run_started = Instant::now();
    let mut track_dump = config
        .track_dump
        .as_ref()
        .map(|path| TrackDump::create(path))
        .transpose()?;

    let mut playback_result: Result<()> = Ok(());
    while playback_result.is_ok() {
        let Some(tracked) = pipeline.next_tracked() else {
            break;
        };

        let queue_depths = pipeline.queue_depths();
        let confirmed_tracks = tracked
            .tracks
            .iter()
            .filter(|track| track.state == TrackState::Confirmed)
            .count();

        if let Some(dump) = track_dump.as_mut()
            && let Err(err) = dump.write_frame(tracked.stamp, &tracked.tracks)
        {
            playback_result = Err(err);
            break;
        }

        let frame_count = tracked.stamp.index + 1;
        let metrics = FrameMetrics {
            timings: tracked.timings,
            // Overlay shows the prior frame's measured render time (or unavailable).
            render_ms: last_render_ms.unwrap_or(f64::NAN),
            queue_depths,
            fps: rolling_fps.displayed_fps(),
            confirmed_tracks,
        };

        let render_start = Instant::now();
        let presentation = render.present(tracked, &metrics);
        let render_ms = render_start.elapsed().as_secs_f64() * 1000.0;
        last_render_ms = Some(render_ms);

        let now = Instant::now();
        rolling_fps.record_frame(now.duration_since(last_iteration_end));
        last_iteration_end = now;

        let metrics = FrameMetrics {
            render_ms,
            fps: rolling_fps.displayed_fps(),
            ..metrics
        };
        run_stats.record(&metrics);
        maybe_log_instrumentation(
            &metrics,
            frame_count,
            run_started,
            &mut last_instrumentation_log,
        );

        match presentation {
            Ok(Presentation::Continue) => {}
            Ok(Presentation::QuitRequested) => break,
            Err(err) => {
                playback_result = Err(err);
                break;
            }
        }
    }

    pipeline.request_shutdown();
    let join_result = pipeline.join();
    if playback_result.is_ok() {
        if let Ok(stats) = join_result {
            log_playback_summary(&stats.decode_summary, stats.rejected_updates);
            log_instrumentation_summary(&run_stats.summary());
            if let Some(dump) = track_dump {
                playback_result = dump.finish();
            }
        } else if let Err(err) = join_result {
            playback_result = Err(err);
        }
    }

    let Err(cleanup_err) = render.close() else {
        return playback_result;
    };

    match playback_result {
        Ok(()) => Err(cleanup_err),
        Err(process_err) => {
            tracing::error!(error = %format!("{cleanup_err:#}"), "display window cleanup failed");
            Err(process_err)
        }
    }
}

fn maybe_log_instrumentation(
    metrics: &FrameMetrics,
    frame_count: u64,
    run_started: Instant,
    last_log: &mut Option<Instant>,
) {
    let now = Instant::now();
    if last_log.is_some_and(|last| now.duration_since(last) < INSTRUMENTATION_LOG_INTERVAL) {
        return;
    }

    tracing::info!(
        elapsed_seconds = run_started.elapsed().as_secs_f64(),
        frame_count,
        decode_ms = metrics.timings.decode_ms,
        preprocess_ms = metrics.timings.preprocess_ms,
        inference_ms = metrics.timings.inference_ms,
        tracking_ms = metrics.timings.tracking_ms,
        render_ms = metrics.render_ms,
        decoded_depth = format_depth(
            metrics.queue_depths.decoded.0,
            metrics.queue_depths.decoded.1
        ),
        prepared_depth = format_depth(
            metrics.queue_depths.prepared.0,
            metrics.queue_depths.prepared.1
        ),
        detected_depth = format_depth(
            metrics.queue_depths.detected.0,
            metrics.queue_depths.detected.1
        ),
        tracked_depth = format_depth(
            metrics.queue_depths.tracked.0,
            metrics.queue_depths.tracked.1
        ),
        "instrumentation progress"
    );
    *last_log = Some(now);
}
