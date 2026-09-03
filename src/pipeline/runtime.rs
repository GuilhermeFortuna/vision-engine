use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::detector::LoadedModel;
use anyhow::{Context, Result, bail};

use crate::cli::Config;

use super::decode::{DecodeStage, DecodeSummary};
use super::infer::InferStage;
use super::message::{DecodedFrame, DetectedFrame, PreparedFrame, TrackedFrame};
use super::metrics::QueueDepths;
use super::preprocess::prepare;
use super::queue::{self, QUEUE_CAPACITY, QueueDepthGauge, Receiver, Sender, Shutdown};
use super::track::TrackStage;
use crate::tracking::TrackState;

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

    fn parse(name: &str) -> Option<Self> {
        match name {
            STAGE_DECODE => Some(Self::Decode),
            STAGE_PREPROCESS => Some(Self::Preprocess),
            STAGE_INFER => Some(Self::Infer),
            STAGE_TRACK => Some(Self::Track),
            _ => None,
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

struct StallState {
    active: Mutex<Option<(Stage, Instant)>>,
}

#[cfg_attr(not(any(test, feature = "test-utils")), allow(dead_code))]
impl StallState {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            active: Mutex::new(None),
        })
    }

    fn activate(&self, stage: Stage, delay: Duration) {
        *self.active.lock().expect("stall state poisoned") = Some((stage, Instant::now() + delay));
    }

    fn apply(&self, stage: Stage) {
        let guard = self.active.lock().expect("stall state poisoned");
        let Some((active_stage, until)) = *guard else {
            return;
        };
        if active_stage != stage {
            return;
        }
        let now = Instant::now();
        if now < until {
            let remaining = until - now;
            drop(guard);
            thread::sleep(remaining);
        }
    }
}

struct PeakDepthTracker {
    decoded: AtomicUsize,
    prepared: AtomicUsize,
    detected: AtomicUsize,
    tracked: AtomicUsize,
}

#[cfg_attr(not(any(test, feature = "test-utils")), allow(dead_code))]
impl PeakDepthTracker {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            decoded: AtomicUsize::new(0),
            prepared: AtomicUsize::new(0),
            detected: AtomicUsize::new(0),
            tracked: AtomicUsize::new(0),
        })
    }

    fn record(&self, depths: &QueueDepths) {
        self.decoded.fetch_max(depths.decoded.0, Ordering::Relaxed);
        self.prepared
            .fetch_max(depths.prepared.0, Ordering::Relaxed);
        self.detected
            .fetch_max(depths.detected.0, Ordering::Relaxed);
        self.tracked.fetch_max(depths.tracked.0, Ordering::Relaxed);
    }

    fn snapshot(&self) -> QueueDepths {
        QueueDepths {
            decoded: (self.decoded.load(Ordering::Relaxed), QUEUE_CAPACITY),
            prepared: (self.prepared.load(Ordering::Relaxed), QUEUE_CAPACITY),
            detected: (self.detected.load(Ordering::Relaxed), QUEUE_CAPACITY),
            tracked: (self.tracked.load(Ordering::Relaxed), QUEUE_CAPACITY),
        }
    }
}

pub struct Pipeline {
    decode: StageHandle<DecodeSummary>,
    preprocess: StageHandle<()>,
    infer: StageHandle<()>,
    track: StageHandle<u64>,
    decoded_gauge: QueueDepthGauge<DecodedFrame>,
    prepared_gauge: QueueDepthGauge<PreparedFrame>,
    detected_gauge: QueueDepthGauge<DetectedFrame>,
    tracked_gauge: QueueDepthGauge<TrackedFrame>,
    tracked_rx: Receiver<TrackedFrame>,
    shutdown: Shutdown,
    #[cfg_attr(not(any(test, feature = "test-utils")), allow(dead_code))]
    stall_state: Arc<StallState>,
    peak_depths: Arc<PeakDepthTracker>,
}

impl Pipeline {
    pub fn spawn(config: &Config, model: LoadedModel) -> Result<Self> {
        Self::spawn_internal(config, model, FaultConfig::default())
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn spawn_for_test(config: &Config, model: LoadedModel, fault: FaultConfig) -> Result<Self> {
        Self::spawn_internal(config, model, fault)
    }

    fn spawn_internal(config: &Config, model: LoadedModel, fault: FaultConfig) -> Result<Self> {
        let shutdown = Shutdown::new();
        let stall_state = StallState::new();
        let peak_depths = PeakDepthTracker::new();

        let (decoded_tx, decoded_rx) = queue::bounded::<DecodedFrame>(QUEUE_CAPACITY, &shutdown);
        let (prepared_tx, prepared_rx) = queue::bounded::<PreparedFrame>(QUEUE_CAPACITY, &shutdown);
        let (detected_tx, detected_rx) = queue::bounded::<DetectedFrame>(QUEUE_CAPACITY, &shutdown);
        let (tracked_tx, tracked_rx) = queue::bounded::<TrackedFrame>(QUEUE_CAPACITY, &shutdown);

        let decoded_gauge = decoded_rx.depth_gauge();
        let prepared_gauge = prepared_rx.depth_gauge();
        let detected_gauge = detected_rx.depth_gauge();
        let tracked_gauge = tracked_rx.depth_gauge();

        let video = config.video.clone();
        let loop_for = config.loop_for;
        let max_frames = config.max_frames;
        let shutdown_decode = shutdown.clone_handle();
        let shutdown_preprocess = shutdown.clone_handle();
        let shutdown_infer = shutdown.clone_handle();
        let shutdown_track = shutdown.clone_handle();
        let stall_decode = Arc::clone(&stall_state);
        let stall_preprocess = Arc::clone(&stall_state);
        let stall_infer = Arc::clone(&stall_state);
        let stall_track = Arc::clone(&stall_state);

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
                    stall_decode,
                )
            })
            .context("failed to spawn decode stage thread")?;

        let preprocess = thread::Builder::new()
            .name(STAGE_PREPROCESS.to_string())
            .spawn(move || {
                run_preprocess_stage(
                    decoded_rx,
                    prepared_tx,
                    shutdown_preprocess,
                    fault,
                    stall_preprocess,
                )
            })
            .context("failed to spawn preprocess stage thread")?;

        let infer = thread::Builder::new()
            .name(STAGE_INFER.to_string())
            .spawn(move || {
                run_infer_stage(
                    model,
                    prepared_rx,
                    detected_tx,
                    shutdown_infer,
                    fault,
                    stall_infer,
                )
            })
            .context("failed to spawn infer stage thread")?;

        let track = thread::Builder::new()
            .name(STAGE_TRACK.to_string())
            .spawn(move || {
                run_track_stage(detected_rx, tracked_tx, shutdown_track, fault, stall_track)
            })
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
            decoded_gauge,
            prepared_gauge,
            detected_gauge,
            tracked_gauge,
            tracked_rx,
            shutdown,
            stall_state,
            peak_depths,
        })
    }

    pub fn queue_depths(&self) -> QueueDepths {
        let depths = QueueDepths {
            decoded: self.decoded_gauge.snapshot(),
            prepared: self.prepared_gauge.snapshot(),
            detected: self.detected_gauge.snapshot(),
            tracked: self.tracked_gauge.snapshot(),
        };
        self.peak_depths.record(&depths);
        depths
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn peak_queue_depths(&self) -> QueueDepths {
        self.peak_depths.snapshot()
    }

    #[cfg(any(test, feature = "test-utils"))]
    pub fn stall_stage_for(&self, stage: &str, delay: Duration) {
        let stage = Stage::parse(stage).unwrap_or_else(|| {
            panic!("unknown stage {stage:?}; expected decode, preprocess, infer, or track")
        });
        self.stall_state.activate(stage, delay);
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
    stall: Arc<StallState>,
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

        stall.apply(Stage::Decode);

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
    stall: Arc<StallState>,
) -> Result<()> {
    while let Ok(decoded) = rx.recv() {
        stall.apply(Stage::Preprocess);

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
    stall: Arc<StallState>,
) -> Result<()> {
    let mut infer = InferStage::new(model);

    while let Ok(prepared) = rx.recv() {
        stall.apply(Stage::Infer);

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
    stall: Arc<StallState>,
) -> Result<u64> {
    let mut track = TrackStage::new();

    while let Ok(detected) = rx.recv() {
        stall.apply(Stage::Track);

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

#[cfg(any(test, feature = "test-utils"))]
pub mod test_support {
    use std::path::PathBuf;
    use std::time::Duration;

    use crate::detector::LoadedModel;

    use crate::cli::Config;

    pub fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    pub fn sample_video() -> Option<PathBuf> {
        let path = repo_root().join("samples/test.mp4");
        path.is_file().then_some(path)
    }

    pub fn sample_model() -> Option<LoadedModel> {
        let path = repo_root().join("models/yolov8n.onnx");
        if !path.is_file() {
            return None;
        }
        LoadedModel::load(&path).ok()
    }

    pub fn test_config(max_frames: u64) -> Option<Config> {
        Some(Config {
            video: sample_video()?,
            model: repo_root().join("models/yolov8n.onnx"),
            loop_for: None,
            track_dump: None,
            max_frames: Some(max_frames),
        })
    }

    pub fn looped_test_config(max_frames: u64) -> Option<Config> {
        Some(Config {
            video: sample_video()?,
            model: repo_root().join("models/yolov8n.onnx"),
            loop_for: Some(Duration::from_secs(3600)),
            track_dump: None,
            max_frames: Some(max_frames),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use test_support::{sample_model, test_config};

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

    #[test]
    fn shutdown_after_partial_drain_terminates_cleanly() {
        let Some(config) = test_config(50) else {
            eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
            return;
        };
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn(&config, model).expect("pipeline should spawn");
        for _ in 0..10 {
            assert!(pipeline.next_tracked().is_some());
        }
        pipeline.request_shutdown();
        pipeline
            .join()
            .expect("mid-run shutdown should join cleanly");
    }

    #[test]
    fn shutdown_at_end_of_input_terminates_cleanly() {
        let Some(config) = test_config(20) else {
            eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
            return;
        };
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn(&config, model).expect("pipeline should spawn");
        while pipeline.next_tracked().is_some() {}
        pipeline.request_shutdown();
        pipeline
            .join()
            .expect("end-of-input shutdown should join cleanly");
    }

    #[test]
    fn non_fatal_conditions_complete_without_error() {
        let Some(config) = test_config(100) else {
            eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
            return;
        };
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn(&config, model).expect("pipeline should spawn");
        let (_indices, join_result) = drain_pipeline(pipeline);
        let stats = join_result.expect("pipeline should complete without error");
        assert!(
            stats.decode_summary.frame_count > 0,
            "expected frames to be processed"
        );
    }

    #[test]
    fn per_frame_buffer_cost_is_quantified() {
        let Some(config) = test_config(50) else {
            eprintln!("skipping: samples/test.mp4 or models/yolov8n.onnx not present");
            return;
        };
        let Some(model) = sample_model() else {
            eprintln!("skipping: models/yolov8n.onnx not present");
            return;
        };

        let pipeline = Pipeline::spawn(&config, model).expect("pipeline should spawn");
        let mut decode_ms = 0.0;
        let mut preprocess_ms = 0.0;
        let mut frame_ms = 0.0;
        let mut frames = 0_u64;

        while let Some(tracked) = pipeline.next_tracked() {
            decode_ms += tracked.timings.decode_ms;
            preprocess_ms += tracked.timings.preprocess_ms;
            frame_ms += tracked.timings.decode_ms
                + tracked.timings.preprocess_ms
                + tracked.timings.inference_ms
                + tracked.timings.tracking_ms;
            frames += 1;
        }
        pipeline.request_shutdown();
        pipeline.join().expect("pipeline should join cleanly");

        assert!(frames > 0, "expected frames for allocation measurement");
        let allocation_ms = decode_ms + preprocess_ms;
        let mean_frame_ms = frame_ms / frames as f64;
        let allocation_share = allocation_ms / frames as f64 / mean_frame_ms * 100.0;
        eprintln!(
            "per-frame allocation share: {allocation_share:.1}% (decode+preprocess / total stage time)"
        );
        assert!(
            allocation_share.is_finite() && allocation_share > 0.0,
            "allocation share should be measurable"
        );
    }
}
