use std::path::PathBuf;
use std::thread::{self, JoinHandle};
use std::time::Duration;

use anyhow::{Context, Result, bail};
use vision_engine::detector::LoadedModel;

use crate::cli::Config;

use super::decode::{DecodeStage, DecodeSummary};
use super::infer::InferStage;
use super::message::{DecodedFrame, DetectedFrame, PreparedFrame, TrackedFrame};
use super::preprocess::prepare;
use super::queue::{self, QUEUE_CAPACITY, Receiver, Sender, Shutdown};
use super::track::TrackStage;
use vision_engine::tracking::TrackState;

const STAGE_DECODE: &str = "decode";
const STAGE_PREPROCESS: &str = "preprocess";
const STAGE_INFER: &str = "infer";
const STAGE_TRACK: &str = "track";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Decode,
    Preprocess,
    Infer,
    Track,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Self::Decode => STAGE_DECODE,
            Self::Preprocess => STAGE_PREPROCESS,
            Self::Infer => STAGE_INFER,
            Self::Track => STAGE_TRACK,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct FaultConfig {
    pub fail_at: Option<(Stage, u64)>,
    pub panic_at: Option<(Stage, u64)>,
}

#[derive(Debug)]
pub struct PipelineRunStats {
    pub decode_summary: DecodeSummary,
    pub rejected_updates: u64,
}

struct StageHandle<T> {
    name: &'static str,
    join: JoinHandle<Result<T>>,
}

pub struct Pipeline {
    decode: StageHandle<DecodeSummary>,
    preprocess: StageHandle<()>,
    infer: StageHandle<()>,
    track: StageHandle<u64>,
    tracked_rx: Receiver<TrackedFrame>,
    shutdown: Shutdown,
}

impl Pipeline {
    pub fn spawn(config: &Config, model: LoadedModel) -> Result<Self> {
        Self::spawn_with_fault(config, model, FaultConfig::default())
    }

    #[cfg(test)]
    pub fn spawn_for_test(config: &Config, model: LoadedModel, fault: FaultConfig) -> Result<Self> {
        Self::spawn_with_fault(config, model, fault)
    }

    fn spawn_with_fault(config: &Config, model: LoadedModel, fault: FaultConfig) -> Result<Self> {
        let shutdown = Shutdown::new();

        let (decoded_tx, decoded_rx) = queue::bounded::<DecodedFrame>(QUEUE_CAPACITY, &shutdown);
        let (prepared_tx, prepared_rx) = queue::bounded::<PreparedFrame>(QUEUE_CAPACITY, &shutdown);
        let (detected_tx, detected_rx) = queue::bounded::<DetectedFrame>(QUEUE_CAPACITY, &shutdown);
        let (tracked_tx, tracked_rx) = queue::bounded::<TrackedFrame>(QUEUE_CAPACITY, &shutdown);

        let video = config.video.clone();
        let loop_for = config.loop_for;
        let max_frames = config.max_frames;
        let shutdown_decode = shutdown.clone_handle();
        let shutdown_preprocess = shutdown.clone_handle();
        let shutdown_infer = shutdown.clone_handle();
        let shutdown_track = shutdown.clone_handle();

        let decode = thread::Builder::new()
            .name(STAGE_DECODE.to_string())
            .spawn(move || {
                run_decode_stage(
                    video,
                    loop_for,
                    max_frames,
                    decoded_tx,
                    shutdown_decode,
                    fault,
                )
            })
            .context("failed to spawn decode stage thread")?;

        let preprocess = thread::Builder::new()
            .name(STAGE_PREPROCESS.to_string())
            .spawn(move || {
                run_preprocess_stage(decoded_rx, prepared_tx, shutdown_preprocess, fault)
            })
            .context("failed to spawn preprocess stage thread")?;

        let infer = thread::Builder::new()
            .name(STAGE_INFER.to_string())
            .spawn(move || run_infer_stage(model, prepared_rx, detected_tx, shutdown_infer, fault))
            .context("failed to spawn infer stage thread")?;

        let track = thread::Builder::new()
            .name(STAGE_TRACK.to_string())
            .spawn(move || run_track_stage(detected_rx, tracked_tx, shutdown_track, fault))
            .context("failed to spawn track stage thread")?;

        Ok(Self {
            decode: StageHandle {
                name: STAGE_DECODE,
                join: decode,
            },
            preprocess: StageHandle {
                name: STAGE_PREPROCESS,
                join: preprocess,
            },
            infer: StageHandle {
                name: STAGE_INFER,
                join: infer,
            },
            track: StageHandle {
                name: STAGE_TRACK,
                join: track,
            },
            tracked_rx,
            shutdown,
        })
    }

    pub fn next_tracked(&self) -> Option<TrackedFrame> {
        self.tracked_rx.recv().ok()
    }

    pub fn request_shutdown(&self) {
        self.shutdown.request();
    }

    pub fn join(self) -> Result<PipelineRunStats> {
        let Self {
            decode,
            preprocess,
            infer,
            track,
            tracked_rx,
            ..
        } = self;
        drop(tracked_rx);

        let mut first_error: Option<anyhow::Error> = None;
        let mut decode_summary = DecodeSummary::default();
        let mut rejected_updates = 0_u64;

        match join_stage(decode.name, decode.join) {
            Ok(summary) => decode_summary = summary,
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                } else {
                    tracing::error!(
                        stage = decode.name,
                        error = %format!("{err:#}"),
                        "stage failed during shutdown"
                    );
                }
            }
        }

        match join_stage(preprocess.name, preprocess.join) {
            Ok(()) => {}
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                } else {
                    tracing::error!(
                        stage = preprocess.name,
                        error = %format!("{err:#}"),
                        "stage failed during shutdown"
                    );
                }
            }
        }

        match join_stage(infer.name, infer.join) {
            Ok(()) => {}
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                } else {
                    tracing::error!(
                        stage = infer.name,
                        error = %format!("{err:#}"),
                        "stage failed during shutdown"
                    );
                }
            }
        }

        match join_stage(track.name, track.join) {
            Ok(count) => rejected_updates = count,
            Err(err) => {
                if first_error.is_none() {
                    first_error = Some(err);
                } else {
                    tracing::error!(
                        stage = track.name,
                        error = %format!("{err:#}"),
                        "stage failed during shutdown"
                    );
                }
            }
        }

        if let Some(err) = first_error {
            return Err(err);
        }

        Ok(PipelineRunStats {
            decode_summary,
            rejected_updates,
        })
    }
}

fn join_stage<T>(name: &'static str, handle: JoinHandle<Result<T>>) -> Result<T> {
    match handle.join() {
        Ok(result) => result,
        Err(panic_payload) => {
            let message = if let Some(message) = panic_payload.downcast_ref::<&str>() {
                (*message).to_string()
            } else if let Some(message) = panic_payload.downcast_ref::<String>() {
                message.clone()
            } else {
                "unknown panic payload".to_string()
            };
            bail!("{name} stage panicked: {message}");
        }
    }
}

fn stage_context(stage: &'static str, index: u64, err: anyhow::Error) -> anyhow::Error {
    err.context(format!("{stage} failed at frame {index}"))
}

fn check_fault(stage: Stage, index: u64, fault: FaultConfig) -> Result<()> {
    if fault.panic_at == Some((stage, index)) {
        panic!("injected panic in {} at frame {}", stage.name(), index);
    }
    if fault.fail_at == Some((stage, index)) {
        bail!("injected failure in {} at frame {}", stage.name(), index);
    }
    Ok(())
}

fn run_decode_stage(
    video: PathBuf,
    loop_for: Option<Duration>,
    max_frames: Option<u64>,
    tx: Sender<DecodedFrame>,
    shutdown: Shutdown,
    fault: FaultConfig,
) -> Result<DecodeSummary> {
    let mut decode = match DecodeStage::open(&video, loop_for, max_frames) {
        Ok(decode) => decode,
        Err(err) => {
            shutdown.request();
            return Err(err.context(format!("{STAGE_DECODE} failed at frame 0")));
        }
    };

    loop {
        if shutdown.is_requested() {
            break;
        }

        let decoded = match decode.next() {
            Ok(Some(frame)) => frame,
            Ok(None) => break,
            Err(err) => {
                shutdown.request();
                let index = decode.summary().frame_count;
                return Err(stage_context(STAGE_DECODE, index, err));
            }
        };

        let index = decoded.stamp.index;
        if let Err(err) = check_fault(Stage::Decode, index, fault) {
            shutdown.request();
            return Err(stage_context(STAGE_DECODE, index, err));
        }

        if tx.send(decoded).is_err() {
            break;
        }
    }

    drop(tx);
    Ok(decode.summary())
}

fn run_preprocess_stage(
    rx: Receiver<DecodedFrame>,
    tx: Sender<PreparedFrame>,
    shutdown: Shutdown,
    fault: FaultConfig,
) -> Result<()> {
    while let Ok(decoded) = rx.recv() {
        let index = decoded.stamp.index;
        if let Err(err) = check_fault(Stage::Preprocess, index, fault) {
            shutdown.request();
            return Err(stage_context(STAGE_PREPROCESS, index, err));
        }

        let prepared = match prepare(decoded) {
            Ok(prepared) => prepared,
            Err(err) => {
                shutdown.request();
                return Err(stage_context(STAGE_PREPROCESS, index, err));
            }
        };

        if tx.send(prepared).is_err() {
            break;
        }
    }

    drop(tx);
    Ok(())
}

fn run_infer_stage(
    model: LoadedModel,
    rx: Receiver<PreparedFrame>,
    tx: Sender<DetectedFrame>,
    shutdown: Shutdown,
    fault: FaultConfig,
) -> Result<()> {
    let mut infer = InferStage::new(model);

    while let Ok(prepared) = rx.recv() {
        let index = prepared.stamp.index;
        if let Err(err) = check_fault(Stage::Infer, index, fault) {
            shutdown.request();
            return Err(stage_context(STAGE_INFER, index, err));
        }

        let detected = match infer.detect(prepared) {
            Ok(detected) => detected,
            Err(err) => {
                shutdown.request();
                return Err(stage_context(STAGE_INFER, index, err));
            }
        };

        if tx.send(detected).is_err() {
            break;
        }
    }

    drop(tx);
    Ok(())
}

fn run_track_stage(
    rx: Receiver<DetectedFrame>,
    tx: Sender<TrackedFrame>,
    shutdown: Shutdown,
    fault: FaultConfig,
) -> Result<u64> {
    let mut track = TrackStage::new();

    while let Ok(detected) = rx.recv() {
        let index = detected.stamp.index;
        if let Err(err) = check_fault(Stage::Track, index, fault) {
            shutdown.request();
            return Err(stage_context(STAGE_TRACK, index, err));
        }

        let tracked = match track.update(detected) {
            Ok(tracked) => tracked,
            Err(err) => {
                shutdown.request();
                return Err(stage_context(STAGE_TRACK, index, err));
            }
        };

        let confirmed_tracks = tracked
            .tracks
            .iter()
            .filter(|track| track.state == TrackState::Confirmed)
            .count();
        track.maybe_log_progress(tracked.stamp, confirmed_tracks);

        if tx.send(tracked).is_err() {
            break;
        }
    }

    drop(tx);
    Ok(track.rejected_updates())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;
    use vision_engine::detector::LoadedModel;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn sample_video() -> Option<PathBuf> {
        let path = repo_root().join("samples/test.mp4");
        path.is_file().then_some(path)
    }

    fn sample_model() -> Option<LoadedModel> {
        let path = repo_root().join("models/yolov8n.onnx");
        if !path.is_file() {
            return None;
        }
        LoadedModel::load(&path).ok()
    }

    fn test_config(max_frames: u64) -> Option<Config> {
        Some(Config {
            video: sample_video()?,
            model: repo_root().join("models/yolov8n.onnx"),
            loop_for: None,
            track_dump: None,
            max_frames: Some(max_frames),
        })
    }

    fn drain_pipeline(pipeline: Pipeline) -> (Vec<u64>, Result<PipelineRunStats>) {
        let mut indices = Vec::new();
        while let Some(tracked) = pipeline.next_tracked() {
            indices.push(tracked.stamp.index);
        }
        pipeline.request_shutdown();
        let join_result = pipeline.join();
        (indices, join_result)
    }

    #[test]
    fn frame_indices_are_contiguous_and_in_order() {
        let Some(config) = test_config(50) else {
            eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
            return;
        };
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn(&config, model).expect("pipeline should spawn");
        let (indices, join_result) = drain_pipeline(pipeline);
        join_result.expect("pipeline should join cleanly");

        let expected: Vec<u64> = (0..50).collect();
        assert_eq!(
            indices, expected,
            "frame indices must be contiguous and in order"
        );
    }

    #[test]
    fn end_of_input_drains_all_frames() {
        let Some(config) = test_config(50) else {
            eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
            return;
        };
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn(&config, model).expect("pipeline should spawn");
        let (indices, join_result) = drain_pipeline(pipeline);
        let stats = join_result.expect("pipeline should join cleanly");

        assert_eq!(
            indices.len(),
            50,
            "renderer should receive every decoded frame"
        );
        assert_eq!(
            stats.decode_summary.frame_count, 50,
            "decoder should produce 50 frames"
        );
    }

    #[test]
    fn induced_failure_reports_stage_and_frame_index() {
        for stage in [Stage::Decode, Stage::Preprocess, Stage::Infer, Stage::Track] {
            let Some(config) = test_config(20) else {
                eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
                return;
            };
            let Some(model) = sample_model() else {
                eprintln!("skipping: models/yolov8n.onnx not present");
                return;
            };

            let fault = FaultConfig {
                fail_at: Some((stage, 10)),
                panic_at: None,
            };
            let pipeline =
                Pipeline::spawn_for_test(&config, model, fault).expect("pipeline should spawn");
            let (_indices, join_result) = drain_pipeline(pipeline);
            let err = join_result.expect_err("pipeline should fail");
            let message = format!("{err:#}");
            assert!(
                message.contains(stage.name()),
                "expected stage name {} in error: {message}",
                stage.name()
            );
            assert!(
                message.contains("frame 10"),
                "expected frame index in error: {message}"
            );
        }
    }

    #[test]
    fn induced_panic_reports_stage_without_hanging() {
        for stage in [Stage::Decode, Stage::Preprocess, Stage::Infer, Stage::Track] {
            let Some(config) = test_config(20) else {
                eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
                return;
            };
            let Some(model) = sample_model() else {
                eprintln!("skipping: models/yolov8n.onnx not present");
                return;
            };

            let fault = FaultConfig {
                fail_at: None,
                panic_at: Some((stage, 10)),
            };
            let pipeline =
                Pipeline::spawn_for_test(&config, model, fault).expect("pipeline should spawn");
            let (_indices, join_result) = drain_pipeline(pipeline);
            let err = join_result.expect_err("pipeline should fail");
            let message = format!("{err:#}");
            assert!(
                message.contains(stage.name()),
                "expected stage name {} in error: {message}",
                stage.name()
            );
            assert!(
                message.contains("panicked"),
                "expected panic report in error: {message}"
            );
        }
    }

    #[test]
    fn shutdown_with_full_queues_terminates_and_joins() {
        let Some(config) = test_config(50) else {
            eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
            return;
        };
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn(&config, model).expect("pipeline should spawn");
        pipeline.request_shutdown();
        let join_result = pipeline.join();
        join_result.expect("shutdown with full queues should join cleanly");
    }

    #[test]
    fn shutdown_before_first_frame_terminates_cleanly() {
        let Some(config) = test_config(50) else {
            eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
            return;
        };
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn(&config, model).expect("pipeline should spawn");
        pipeline.request_shutdown();
        let join_result = pipeline.join();
        join_result.expect("early shutdown should join cleanly");
    }
}
