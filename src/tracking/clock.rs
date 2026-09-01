const NOMINAL_FPS: f64 = 30.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TimeSource {
    Reported,
    DerivedFromFrameRate,
    DerivedFromIndex,
}

#[derive(Clone, Copy, PartialEq, Debug)]
pub struct FrameStamp {
    pub index: u64,
    pub media_ms: f64,
    pub source: TimeSource,
    pub adjusted: bool,
}

pub struct FrameClock {
    index: u64,
    last_media_ms: f64,
    source_fps: Option<f64>,
    adjustments: u64,
}

impl FrameClock {
    pub fn new(source_fps: Option<f64>) -> Self {
        Self {
            index: 0,
            last_media_ms: 0.0,
            source_fps: sanitize_fps(source_fps),
            adjustments: 0,
        }
    }

    pub fn stamp(&mut self, reported_ms: Option<f64>) -> FrameStamp {
        let index = self.index;
        let (mut media_ms, source) = resolve_media_ms(index, reported_ms, self.source_fps);
        let mut adjusted = false;

        if index > 0 && media_ms < self.last_media_ms {
            media_ms = self.last_media_ms;
            adjusted = true;
            self.adjustments += 1;
        }

        self.last_media_ms = media_ms;
        self.index += 1;

        FrameStamp {
            index,
            media_ms,
            source,
            adjusted,
        }
    }

    pub fn adjustments(&self) -> u64 {
        self.adjustments
    }

    pub fn stamped_count(&self) -> u64 {
        self.index
    }
}

fn sanitize_fps(fps: Option<f64>) -> Option<f64> {
    match fps {
        Some(value) if value.is_finite() && value > 0.0 => Some(value),
        _ => None,
    }
}

fn sanitize_reported_ms(reported_ms: Option<f64>) -> Option<f64> {
    match reported_ms {
        Some(value) if value.is_finite() && value >= 0.0 => Some(value),
        _ => None,
    }
}

fn resolve_media_ms(
    index: u64,
    reported_ms: Option<f64>,
    source_fps: Option<f64>,
) -> (f64, TimeSource) {
    if let Some(media_ms) = sanitize_reported_ms(reported_ms) {
        return (media_ms, TimeSource::Reported);
    }

    if let Some(fps) = source_fps {
        let media_ms = index as f64 / fps * 1000.0;
        return (media_ms, TimeSource::DerivedFromFrameRate);
    }

    let media_ms = index as f64 * (1000.0 / NOMINAL_FPS);
    (media_ms, TimeSource::DerivedFromIndex)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_uses_reported_time_verbatim() {
        let mut clock = FrameClock::new(Some(30.0));
        let stamp = clock.stamp(Some(123.4));
        assert_eq!(stamp.index, 0);
        assert!((stamp.media_ms - 123.4).abs() < f64::EPSILON);
        assert_eq!(stamp.source, TimeSource::Reported);
        assert!(!stamp.adjusted);
    }

    #[test]
    fn stamp_falls_back_to_frame_rate_when_report_missing() {
        let mut clock = FrameClock::new(Some(25.0));
        let stamp = clock.stamp(None);
        assert_eq!(stamp.index, 0);
        assert!((stamp.media_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(stamp.source, TimeSource::DerivedFromFrameRate);

        let stamp = clock.stamp(None);
        assert_eq!(stamp.index, 1);
        assert!((stamp.media_ms - 40.0).abs() < f64::EPSILON);
        assert_eq!(stamp.source, TimeSource::DerivedFromFrameRate);
    }

    #[test]
    fn stamp_falls_back_to_index_when_frame_rate_missing() {
        let mut clock = FrameClock::new(None);
        let stamp = clock.stamp(None);
        assert_eq!(stamp.index, 0);
        assert!((stamp.media_ms - 0.0).abs() < f64::EPSILON);
        assert_eq!(stamp.source, TimeSource::DerivedFromIndex);

        let stamp = clock.stamp(None);
        assert_eq!(stamp.index, 1);
        assert!((stamp.media_ms - (1000.0 / NOMINAL_FPS)).abs() < f64::EPSILON);
        assert_eq!(stamp.source, TimeSource::DerivedFromIndex);
    }

    #[test]
    fn stamp_clamps_regression_and_counts_adjustment() {
        let mut clock = FrameClock::new(Some(30.0));
        let first = clock.stamp(Some(100.0));
        assert!(!first.adjusted);

        let second = clock.stamp(Some(50.0));
        assert!(second.adjusted);
        assert!((second.media_ms - 100.0).abs() < f64::EPSILON);
        assert_eq!(clock.adjustments(), 1);
    }

    #[test]
    fn stamp_increments_index_by_one() {
        let mut clock = FrameClock::new(None);
        assert_eq!(clock.stamp(None).index, 0);
        assert_eq!(clock.stamp(None).index, 1);
        assert_eq!(clock.stamp(None).index, 2);
    }
}
