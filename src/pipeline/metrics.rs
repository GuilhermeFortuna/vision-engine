use std::time::Duration;

const MIN_FPS_WINDOW: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, Copy)]
pub struct FrameMetrics {
    pub decode_ms: f64,
    pub inference_ms: f64,
    pub tracking_ms: f64,
    pub fps: Option<f64>,
    pub confirmed_tracks: usize,
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

#[cfg(test)]
mod tests {
    use super::*;

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
