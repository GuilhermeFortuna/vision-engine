mod decode;
mod infer;
mod metrics;
mod preprocess;
mod render;
mod track;
mod track_dump;

use std::time::Instant;

use anyhow::Result;
use opencv::prelude::*;
use vision_engine::detector::LoadedModel;
use vision_engine::tracking::TrackState;

use crate::cli::Config;

use decode::{DecodeOutcome, DecodeStage, log_playback_summary};
use infer::InferStage;
use metrics::{FrameMetrics, RollingFps};
use preprocess::prepare;
use render::{Presentation, RenderStage};
use track::TrackStage;
use track_dump::TrackDump;

pub fn run(config: &Config) -> Result<()> {
    let model = LoadedModel::load(&config.model)?;
    let mut infer_stage = InferStage::new(model);
    let mut decode = DecodeStage::open(&config.video, config.loop_for, config.max_frames)?;
    let mut render = RenderStage::open()?;
    let mut track_stage = TrackStage::new();
    let mut rolling_fps = RollingFps::new();
    let mut last_iteration_end = Instant::now();
    let mut track_dump = config
        .track_dump
        .as_ref()
        .map(|path| TrackDump::create(path))
        .transpose()?;

    let playback_result = (|| -> Result<()> {
        let mut frame = Mat::default();

        loop {
            match decode.next_into(&mut frame)? {
                DecodeOutcome::EndOfRun => break,
                DecodeOutcome::Frame { stamp, decode_ms } => {
                    let prepared = prepare(&frame)?;
                    let detected = infer_stage.detect(&prepared)?;
                    let tracked = track_stage.update(&detected.detections, stamp)?;

                    let now = Instant::now();
                    rolling_fps.record_frame(now.duration_since(last_iteration_end));
                    last_iteration_end = now;

                    let confirmed_tracks = tracked
                        .tracks
                        .iter()
                        .filter(|track| track.state == TrackState::Confirmed)
                        .count();
                    track_stage.maybe_log_progress(stamp, confirmed_tracks);

                    if let Some(dump) = track_dump.as_mut() {
                        dump.write_frame(stamp, &tracked.tracks)?;
                    }

                    let metrics = FrameMetrics {
                        decode_ms,
                        inference_ms: detected.inference_ms,
                        tracking_ms: tracked.tracking_ms,
                        fps: rolling_fps.displayed_fps(),
                        confirmed_tracks,
                    };

                    match render.present(&mut frame, &tracked.tracks, &metrics)? {
                        Presentation::Continue => {}
                        Presentation::QuitRequested => break,
                    }
                }
            }
        }

        let summary = decode.summary();
        log_playback_summary(&summary, track_stage.rejected_updates());
        if let Some(dump) = track_dump {
            dump.finish()?;
        }
        Ok(())
    })();

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
