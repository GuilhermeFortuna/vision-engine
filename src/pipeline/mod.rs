mod decode;
mod infer;
mod message;
mod metrics;
mod preprocess;
mod queue;
mod render;
mod runtime;
mod track;
mod track_dump;

use std::time::Instant;

use anyhow::Result;
use vision_engine::detector::LoadedModel;
use vision_engine::tracking::TrackState;

use crate::cli::Config;

use decode::log_playback_summary;
use metrics::{FrameMetrics, RollingFps};
use render::{Presentation, RenderStage};
use runtime::Pipeline;
use track_dump::TrackDump;

pub fn run(config: &Config) -> Result<()> {
    let model = LoadedModel::load(&config.model)?;
    let mut render = RenderStage::open()?;
    let pipeline = Pipeline::spawn(config, model)?;
    let mut rolling_fps = RollingFps::new();
    let mut last_iteration_end = Instant::now();
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

        let now = Instant::now();
        rolling_fps.record_frame(now.duration_since(last_iteration_end));
        last_iteration_end = now;

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

        let metrics = FrameMetrics {
            decode_ms: tracked.timings.decode_ms,
            inference_ms: tracked.timings.inference_ms,
            tracking_ms: tracked.timings.tracking_ms,
            fps: rolling_fps.displayed_fps(),
            confirmed_tracks,
        };

        match render.present(tracked, &metrics) {
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
