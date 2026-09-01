use std::fmt;

use crate::tracking::clock::FrameStamp;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

impl BBox {
    pub fn from_center_size(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        let half_w = w / 2.0;
        let half_h = h / 2.0;
        Self {
            x_min: cx - half_w,
            y_min: cy - half_h,
            x_max: cx + half_w,
            y_max: cy + half_h,
        }
    }

    pub fn width(&self) -> f32 {
        self.x_max - self.x_min
    }

    pub fn height(&self) -> f32 {
        self.y_max - self.y_min
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(&self) -> (f32, f32) {
        (
            (self.x_min + self.x_max) / 2.0,
            (self.y_min + self.y_max) / 2.0,
        )
    }

    pub fn aspect_ratio(&self) -> f32 {
        self.width() / self.height()
    }

    pub fn is_valid(&self) -> bool {
        self.x_min.is_finite()
            && self.y_min.is_finite()
            && self.x_max.is_finite()
            && self.y_max.is_finite()
            && self.x_max > self.x_min
            && self.y_max > self.y_min
    }

    pub fn clamped_to(&self, frame_w: f32, frame_h: f32) -> Option<Self> {
        let clamped = Self {
            x_min: self.x_min.clamp(0.0, frame_w),
            y_min: self.y_min.clamp(0.0, frame_h),
            x_max: self.x_max.clamp(0.0, frame_w),
            y_max: self.y_max.clamp(0.0, frame_h),
        };

        if clamped.is_valid() {
            Some(clamped)
        } else {
            None
        }
    }

    pub fn iou(&self, other: &Self) -> f32 {
        let inter_x_min = self.x_min.max(other.x_min);
        let inter_y_min = self.y_min.max(other.y_min);
        let inter_x_max = self.x_max.min(other.x_max);
        let inter_y_max = self.y_max.min(other.y_max);

        let inter_w = (inter_x_max - inter_x_min).max(0.0);
        let inter_h = (inter_y_max - inter_y_min).max(0.0);
        let intersection = inter_w * inter_h;

        if intersection <= 0.0 {
            return 0.0;
        }

        let area_a = self.area();
        let area_b = other.area();
        let union = area_a + area_b - intersection;

        if union <= 0.0 {
            return 0.0;
        }

        intersection / union
    }
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct TrackId(pub u64);

impl fmt::Display for TrackId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "#{}", self.0)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Lost,
}

#[derive(Clone, PartialEq, Debug)]
pub struct Track {
    pub id: TrackId,
    pub class_id: u32,
    pub state: TrackState,
    pub bbox: BBox,
    pub confidence: f32,
    pub first_seen: FrameStamp,
    pub last_seen: FrameStamp,
    pub hits: u32,
    pub misses: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_center_size_round_trips_center_and_size() {
        let bbox = BBox::from_center_size(320.0, 240.0, 100.0, 80.0);
        let (cx, cy) = bbox.center();
        assert!((cx - 320.0).abs() < f32::EPSILON);
        assert!((cy - 240.0).abs() < f32::EPSILON);
        assert!((bbox.width() - 100.0).abs() < f32::EPSILON);
        assert!((bbox.height() - 80.0).abs() < f32::EPSILON);
    }

    #[test]
    fn area_and_aspect_ratio_are_correct() {
        let bbox = BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 40.0,
            y_max: 20.0,
        };
        assert!((bbox.area() - 800.0).abs() < f32::EPSILON);
        assert!((bbox.aspect_ratio() - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn is_valid_rejects_non_finite_and_zero_extent_boxes() {
        assert!(
            !BBox {
                x_min: f32::NAN,
                y_min: 0.0,
                x_max: 10.0,
                y_max: 10.0,
            }
            .is_valid()
        );
        assert!(
            !BBox {
                x_min: 5.0,
                y_min: 5.0,
                x_max: 5.0,
                y_max: 10.0,
            }
            .is_valid()
        );
        assert!(
            BBox {
                x_min: 0.0,
                y_min: 0.0,
                x_max: 10.0,
                y_max: 11.0,
            }
            .is_valid()
        );
    }

    #[test]
    fn clamped_to_returns_none_when_box_collapses() {
        let bbox = BBox {
            x_min: -10.0,
            y_min: -10.0,
            x_max: -5.0,
            y_max: -5.0,
        };
        assert!(bbox.clamped_to(100.0, 100.0).is_none());
    }

    #[test]
    fn iou_identical_boxes_is_one() {
        let box_a = BBox {
            x_min: 10.0,
            y_min: 10.0,
            x_max: 50.0,
            y_max: 50.0,
        };
        assert!((box_a.iou(&box_a) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        let box_a = BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 10.0,
            y_max: 10.0,
        };
        let box_b = BBox {
            x_min: 20.0,
            y_min: 20.0,
            x_max: 30.0,
            y_max: 30.0,
        };
        assert!(box_a.iou(&box_b).abs() < f32::EPSILON);
    }

    #[test]
    fn iou_partial_overlap_is_between_zero_and_one() {
        let box_a = BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 20.0,
            y_max: 20.0,
        };
        let box_b = BBox {
            x_min: 10.0,
            y_min: 10.0,
            x_max: 30.0,
            y_max: 30.0,
        };
        let iou = box_a.iou(&box_b);
        assert!(iou > 0.0 && iou < 1.0);
    }

    #[test]
    fn track_id_displays_as_hash_number() {
        assert_eq!(TrackId(42).to_string(), "#42");
    }

    #[test]
    fn track_id_compares_and_sorts() {
        let mut ids = [TrackId(10), TrackId(2), TrackId(5)];
        ids.sort();
        assert_eq!(ids, [TrackId(2), TrackId(5), TrackId(10)]);
    }
}
