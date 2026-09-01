use ndarray::Array4;
use opencv::core::Mat;
use vision_engine::detector::{Detection, LetterboxTransform};
use vision_engine::tracking::{FrameStamp, Track};

#[derive(Debug, Clone, Copy, Default)]
pub struct StageTimings {
    pub decode_ms: f64,
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub tracking_ms: f64,
}

#[derive(Debug)]
pub struct DecodedFrame {
    pub frame: Mat,
    pub stamp: FrameStamp,
    pub timings: StageTimings,
}

#[derive(Debug)]
pub struct PreparedFrame {
    pub frame: Mat,
    pub stamp: FrameStamp,
    pub timings: StageTimings,
    pub input: Array4<f32>,
    pub transform: LetterboxTransform,
}

#[derive(Debug)]
pub struct DetectedFrame {
    pub frame: Mat,
    pub stamp: FrameStamp,
    pub timings: StageTimings,
    pub detections: Vec<Detection>,
}

#[derive(Debug)]
pub struct TrackedFrame {
    pub frame: Mat,
    pub stamp: FrameStamp,
    pub timings: StageTimings,
    pub tracks: Vec<Track>,
}

#[cfg(test)]
mod tests {
    fn assert_send<T: Send>() {}

    use super::*;

    #[test]
    fn messages_move_between_threads() {
        assert_send::<DecodedFrame>();
        assert_send::<PreparedFrame>();
        assert_send::<DetectedFrame>();
        assert_send::<TrackedFrame>();
    }
}
