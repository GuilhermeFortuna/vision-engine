use nalgebra::{SMatrix, SVector};

use crate::tracking::BBox;

const MIN_AREA: f32 = 1.0;
const POSITION_UNCERTAINTY: f32 = 10.0;
const VELOCITY_UNCERTAINTY_SCALE: f32 = 1000.0;
const PROCESS_NOISE_VELOCITY: f32 = 0.01;
const PROCESS_NOISE_AREA_VELOCITY: f32 = 0.01;
const MEASUREMENT_NOISE_SHAPE: f32 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectReason {
    SingularCovariance,
    NonFiniteState,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UpdateOutcome {
    Applied,
    Rejected(RejectReason),
}

pub struct KalmanBoxTracker {
    x: SVector<f32, 7>,
    p: SMatrix<f32, 7, 7>,
    f: SMatrix<f32, 7, 7>,
    h: SMatrix<f32, 4, 7>,
    q: SMatrix<f32, 7, 7>,
    r: SMatrix<f32, 4, 4>,
    rejected_updates: u64,
}

impl KalmanBoxTracker {
    pub fn new(bbox: &BBox) -> Self {
        let mut x = SVector::<f32, 7>::zeros();
        let (cx, cy) = bbox.center();
        x[0] = cx;
        x[1] = cy;
        x[2] = bbox.area();
        x[3] = bbox.aspect_ratio();

        let mut p = SMatrix::<f32, 7, 7>::identity() * POSITION_UNCERTAINTY;
        p[(4, 4)] = POSITION_UNCERTAINTY * VELOCITY_UNCERTAINTY_SCALE;
        p[(5, 5)] = POSITION_UNCERTAINTY * VELOCITY_UNCERTAINTY_SCALE;
        p[(6, 6)] = POSITION_UNCERTAINTY * VELOCITY_UNCERTAINTY_SCALE;

        Self {
            x,
            p,
            f: motion_matrix(),
            h: observation_matrix(),
            q: process_noise_matrix(),
            r: measurement_noise_matrix(),
            rejected_updates: 0,
        }
    }

    pub fn predict(&mut self) -> BBox {
        self.x = self.f * self.x;
        self.p = self.f * self.p * self.f.transpose() + self.q;

        if self.x[2] <= MIN_AREA {
            self.x[2] = MIN_AREA;
            self.x[6] = 0.0;
        }

        self.bbox()
    }

    pub fn update(&mut self, bbox: &BBox) -> UpdateOutcome {
        if !bbox.is_valid() {
            self.rejected_updates += 1;
            return UpdateOutcome::Rejected(RejectReason::NonFiniteState);
        }

        let x_before = self.x;
        let p_before = self.p;

        let z = bbox_to_measurement(bbox);
        let innovation = z - self.h * self.x;
        let m = self.p * self.h.transpose();
        let innovation_covariance = self.h * self.p * self.h.transpose() + self.r;

        let gain = match innovation_covariance.lu().solve(&m.transpose()) {
            Some(gain_transpose) => gain_transpose.transpose(),
            None => {
                self.rejected_updates += 1;
                return UpdateOutcome::Rejected(RejectReason::SingularCovariance);
            }
        };

        self.x += gain * innovation;
        let identity = SMatrix::<f32, 7, 7>::identity();
        self.p = (identity - gain * self.h) * self.p;

        if !state_is_finite(&self.x, &self.p) {
            self.x = x_before;
            self.p = p_before;
            self.rejected_updates += 1;
            return UpdateOutcome::Rejected(RejectReason::NonFiniteState);
        }

        UpdateOutcome::Applied
    }

    pub fn bbox(&self) -> BBox {
        state_to_bbox(&self.x)
    }

    pub fn rejected_updates(&self) -> u64 {
        self.rejected_updates
    }
}

#[cfg(test)]
impl KalmanBoxTracker {
    pub fn with_state_for_test(x: SVector<f32, 7>, p: SMatrix<f32, 7, 7>) -> Self {
        Self {
            x,
            p,
            f: motion_matrix(),
            h: observation_matrix(),
            q: process_noise_matrix(),
            r: measurement_noise_matrix(),
            rejected_updates: 0,
        }
    }

    pub fn set_covariance_for_test(&mut self, p: SMatrix<f32, 7, 7>) {
        self.p = p;
    }

    pub fn set_measurement_noise_for_test(&mut self, r: SMatrix<f32, 4, 4>) {
        self.r = r;
    }

    pub fn covariance_trace(&self) -> f32 {
        self.p.trace()
    }
}

fn motion_matrix() -> SMatrix<f32, 7, 7> {
    let mut f = SMatrix::<f32, 7, 7>::identity();
    f[(0, 4)] = 1.0;
    f[(1, 5)] = 1.0;
    f[(2, 6)] = 1.0;
    f
}

fn observation_matrix() -> SMatrix<f32, 4, 7> {
    let mut h = SMatrix::<f32, 4, 7>::zeros();
    h[(0, 0)] = 1.0;
    h[(1, 1)] = 1.0;
    h[(2, 2)] = 1.0;
    h[(3, 3)] = 1.0;
    h
}

fn process_noise_matrix() -> SMatrix<f32, 7, 7> {
    let mut q = SMatrix::<f32, 7, 7>::identity();
    q[(4, 4)] = PROCESS_NOISE_VELOCITY;
    q[(5, 5)] = PROCESS_NOISE_VELOCITY;
    q[(6, 6)] = PROCESS_NOISE_VELOCITY * PROCESS_NOISE_AREA_VELOCITY;
    q
}

fn measurement_noise_matrix() -> SMatrix<f32, 4, 4> {
    let mut r = SMatrix::<f32, 4, 4>::identity();
    r[(2, 2)] = MEASUREMENT_NOISE_SHAPE;
    r[(3, 3)] = MEASUREMENT_NOISE_SHAPE;
    r
}

fn bbox_to_measurement(bbox: &BBox) -> SVector<f32, 4> {
    let (cx, cy) = bbox.center();
    SVector::<f32, 4>::new(cx, cy, bbox.area(), bbox.aspect_ratio())
}

fn state_to_bbox(x: &SVector<f32, 7>) -> BBox {
    let cx = x[0];
    let cy = x[1];
    let area = x[2];
    let aspect_ratio = x[3];
    let width = (area * aspect_ratio).sqrt();

    if !width.is_finite() || width <= 0.0 {
        let side = MIN_AREA.sqrt();
        return BBox::from_center_size(cx, cy, side, side);
    }

    let height = area / width;
    if !height.is_finite() || height <= 0.0 {
        let side = MIN_AREA.sqrt();
        return BBox::from_center_size(cx, cy, side, side);
    }

    BBox::from_center_size(cx, cy, width, height)
}

fn state_is_finite(x: &SVector<f32, 7>, p: &SMatrix<f32, 7, 7>) -> bool {
    x.iter().all(|value| value.is_finite()) && p.iter().all(|value| value.is_finite())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROUND_TRIP_TOLERANCE: f32 = 1e-3;
    const PREDICTION_POSITION_TOLERANCE: f32 = 5.0;
    const STATIONARY_POSITION_TOLERANCE: f32 = 1.0;

    fn assert_relative_close(actual: f32, expected: f32, tolerance: f32) {
        let denominator = expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() / denominator <= tolerance,
            "expected {expected}, got {actual}"
        );
    }

    fn assert_bbox_round_trip(original: BBox) {
        let tracker = KalmanBoxTracker::new(&original);
        let restored = tracker.bbox();
        let (orig_cx, orig_cy) = original.center();

        assert_relative_close(restored.center().0, orig_cx, ROUND_TRIP_TOLERANCE);
        assert_relative_close(restored.center().1, orig_cy, ROUND_TRIP_TOLERANCE);
        assert_relative_close(restored.area(), original.area(), ROUND_TRIP_TOLERANCE);
        assert_relative_close(
            restored.aspect_ratio(),
            original.aspect_ratio(),
            ROUND_TRIP_TOLERANCE,
        );
    }

    #[test]
    fn bbox_round_trip_landscape() {
        assert_bbox_round_trip(BBox::from_center_size(200.0, 150.0, 100.0, 50.0));
    }

    #[test]
    fn bbox_round_trip_portrait() {
        assert_bbox_round_trip(BBox::from_center_size(200.0, 150.0, 50.0, 100.0));
    }

    #[test]
    fn bbox_round_trip_square() {
        assert_bbox_round_trip(BBox::from_center_size(200.0, 150.0, 80.0, 80.0));
    }

    #[test]
    fn constant_velocity_prediction_is_accurate() {
        let width = 40.0;
        let height = 30.0;
        let mut tracker =
            KalmanBoxTracker::new(&BBox::from_center_size(100.0, 100.0, width, height));

        for frame in 1..=8 {
            tracker.predict();
            let cx = 100.0 + frame as f32 * 5.0;
            let cy = 100.0 + frame as f32 * 3.0;
            assert_eq!(
                tracker.update(&BBox::from_center_size(cx, cy, width, height)),
                UpdateOutcome::Applied
            );
        }

        let predicted = tracker.predict();
        let (predicted_cx, predicted_cy) = predicted.center();
        let expected_cx = 100.0 + 9.0 * 5.0;
        let expected_cy = 100.0 + 9.0 * 3.0;

        assert!(
            (predicted_cx - expected_cx).abs() <= PREDICTION_POSITION_TOLERANCE,
            "expected cx {expected_cx}, got {predicted_cx}"
        );
        assert!(
            (predicted_cy - expected_cy).abs() <= PREDICTION_POSITION_TOLERANCE,
            "expected cy {expected_cy}, got {predicted_cy}"
        );
    }

    #[test]
    fn stationary_box_remains_stationary() {
        let bbox = BBox::from_center_size(320.0, 240.0, 60.0, 40.0);
        let (origin_cx, origin_cy) = bbox.center();
        let mut tracker = KalmanBoxTracker::new(&bbox);

        for _ in 0..20 {
            tracker.predict();
            assert_eq!(tracker.update(&bbox), UpdateOutcome::Applied);
        }

        let (final_cx, final_cy) = tracker.bbox().center();
        assert!((final_cx - origin_cx).abs() <= STATIONARY_POSITION_TOLERANCE);
        assert!((final_cy - origin_cy).abs() <= STATIONARY_POSITION_TOLERANCE);
    }

    #[test]
    fn covariance_trace_grows_with_prediction_and_shrinks_on_update() {
        let bbox = BBox::from_center_size(100.0, 100.0, 50.0, 40.0);
        let mut tracker = KalmanBoxTracker::new(&bbox);

        let trace_0 = tracker.covariance_trace();
        tracker.predict();
        let trace_1 = tracker.covariance_trace();
        tracker.predict();
        let trace_2 = tracker.covariance_trace();
        tracker.predict();
        let trace_3 = tracker.covariance_trace();

        assert!(trace_1 > trace_0);
        assert!(trace_2 > trace_1);
        assert!(trace_3 > trace_2);

        assert_eq!(tracker.update(&bbox), UpdateOutcome::Applied);
        let trace_after_update = tracker.covariance_trace();
        assert!(trace_after_update < trace_3);
    }

    #[test]
    fn singular_innovation_covariance_is_rejected_without_mutation() {
        let bbox = BBox::from_center_size(100.0, 100.0, 50.0, 40.0);
        let mut tracker = KalmanBoxTracker::new(&bbox);
        tracker.predict();

        let predicted = tracker.bbox();
        tracker.set_covariance_for_test(SMatrix::<f32, 7, 7>::zeros());
        tracker.set_measurement_noise_for_test(SMatrix::<f32, 4, 4>::zeros());

        assert_eq!(
            tracker.update(&bbox),
            UpdateOutcome::Rejected(RejectReason::SingularCovariance)
        );
        assert_eq!(tracker.bbox(), predicted);
        assert_eq!(tracker.rejected_updates(), 1);
    }

    #[test]
    fn non_finite_input_is_rejected_without_mutation() {
        let bbox = BBox::from_center_size(100.0, 100.0, 50.0, 40.0);
        let mut tracker = KalmanBoxTracker::new(&bbox);
        tracker.predict();

        let predicted = tracker.bbox();
        let invalid = BBox {
            x_min: f32::NAN,
            y_min: 0.0,
            x_max: 10.0,
            y_max: 10.0,
        };

        assert_eq!(
            tracker.update(&invalid),
            UpdateOutcome::Rejected(RejectReason::NonFiniteState)
        );
        assert_eq!(tracker.bbox(), predicted);
        assert_eq!(tracker.rejected_updates(), 1);
    }

    #[test]
    fn non_finite_post_update_state_is_rolled_back() {
        let mut x = SVector::<f32, 7>::zeros();
        x[0] = 100.0;
        x[1] = 100.0;
        x[2] = 2000.0;
        x[3] = 1.25;

        let mut p = SMatrix::<f32, 7, 7>::identity();
        p[(0, 0)] = f32::NAN;

        let mut tracker = KalmanBoxTracker::with_state_for_test(x, p);
        let before = tracker.bbox();
        let bbox = BBox::from_center_size(100.0, 100.0, 50.0, 40.0);

        assert_eq!(
            tracker.update(&bbox),
            UpdateOutcome::Rejected(RejectReason::NonFiniteState)
        );
        assert_eq!(tracker.bbox(), before);
        assert_eq!(tracker.rejected_updates(), 1);
    }

    #[test]
    fn long_run_prediction_keeps_boxes_well_formed() {
        let bbox = BBox::from_center_size(200.0, 200.0, 80.0, 60.0);
        let mut x = SVector::<f32, 7>::zeros();
        let (cx, cy) = bbox.center();
        x[0] = cx;
        x[1] = cy;
        x[2] = bbox.area();
        x[3] = bbox.aspect_ratio();
        x[6] = -5.0;

        let mut tracker = KalmanBoxTracker::with_state_for_test(
            x,
            SMatrix::<f32, 7, 7>::identity() * POSITION_UNCERTAINTY,
        );

        for _ in 0..50 {
            let predicted = tracker.predict();
            assert!(predicted.is_valid());
            assert!(predicted.area() > 0.0);
            assert!(predicted.x_min.is_finite());
            assert!(predicted.y_min.is_finite());
            assert!(predicted.x_max.is_finite());
            assert!(predicted.y_max.is_finite());
        }
    }
}
