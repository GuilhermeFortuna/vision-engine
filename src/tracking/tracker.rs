use anyhow::{Result, bail};

use crate::detector::Detection;
use crate::tracking::assignment::associate;
use crate::tracking::clock::FrameStamp;
use crate::tracking::kalman::{KalmanBoxTracker, UpdateOutcome};
use crate::tracking::params::{ASSOCIATION_IOU_GATE, TRACK_PROMOTION_HITS, TRACK_RETENTION_MS};
use crate::tracking::track::{BBox, Track, TrackId, TrackState};

struct TrackEntry {
    track: Track,
    filter: KalmanBoxTracker,
}

pub struct Tracker {
    next_id: u64,
    entries: Vec<TrackEntry>,
    rejected_updates: u64,
    #[cfg(test)]
    test_force_reject_update_for: Option<TrackId>,
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}

impl Tracker {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            entries: Vec::new(),
            rejected_updates: 0,
            #[cfg(test)]
            test_force_reject_update_for: None,
        }
    }

    pub fn live_track_count(&self) -> usize {
        self.entries.len()
    }

    pub fn confirmed_track_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|entry| entry.track.state == TrackState::Confirmed)
            .count()
    }

    pub fn rejected_updates(&self) -> u64 {
        self.rejected_updates
    }

    /// Validates input that originates outside the tracker before updating state.
    /// Rejected Kalman updates remain recoverable operating conditions and are
    /// reported from `update` as diagnostics rather than errors.
    pub fn try_update(
        &mut self,
        detections: &[Detection],
        stamp: FrameStamp,
    ) -> Result<Vec<Track>> {
        if !stamp.media_ms.is_finite() || stamp.media_ms < 0.0 {
            bail!(
                "tracking timestamp unusable at frame {}: media time must be finite and non-negative",
                stamp.index
            );
        }

        for (detection_index, detection) in detections.iter().enumerate() {
            if !detection.confidence.is_finite() || !detection.bbox.is_valid() {
                bail!(
                    "tracking association failed at frame {}: invalid detection at index {detection_index}",
                    stamp.index
                );
            }
        }

        Ok(self.update(detections, stamp))
    }

    pub fn update(&mut self, detections: &[Detection], stamp: FrameStamp) -> Vec<Track> {
        self.entries
            .retain(|entry| stamp.media_ms - entry.track.last_seen.media_ms <= TRACK_RETENTION_MS);

        let predicted_boxes: Vec<BBox> = self
            .entries
            .iter_mut()
            .map(|entry| entry.filter.predict())
            .collect();

        let association_input: Vec<(u32, BBox)> = self
            .entries
            .iter()
            .zip(predicted_boxes.iter())
            .map(|(entry, predicted_box)| (entry.track.class_id, *predicted_box))
            .collect();

        let association = associate(&association_input, detections, ASSOCIATION_IOU_GATE);

        let mut matched_entries = vec![false; self.entries.len()];
        let mut matched_detections = vec![false; detections.len()];

        for (track_idx, detection_idx) in &association.matches {
            matched_entries[*track_idx] = true;
            matched_detections[*detection_idx] = true;

            let entry = &mut self.entries[*track_idx];
            let detection = &detections[*detection_idx];
            let predicted_box = predicted_boxes[*track_idx];

            #[cfg(test)]
            if self.test_force_reject_update_for == Some(entry.track.id) {
                use nalgebra::SMatrix;
                entry
                    .filter
                    .set_covariance_for_test(SMatrix::<f32, 7, 7>::zeros());
                entry
                    .filter
                    .set_measurement_noise_for_test(SMatrix::<f32, 4, 4>::zeros());
                self.test_force_reject_update_for = None;
            }

            match entry.filter.update(&detection.bbox) {
                UpdateOutcome::Applied => {
                    entry.track.bbox = entry.filter.bbox();
                }
                UpdateOutcome::Rejected(reason) => {
                    entry.track.bbox = predicted_box;
                    self.rejected_updates += 1;
                    tracing::warn!(
                        frame_index = stamp.index,
                        ?reason,
                        track_id = entry.track.id.0,
                        "tracking filter update rejected; retaining predicted state"
                    );
                }
            }

            entry.track.confidence = detection.confidence;
            entry.track.last_seen = stamp;
            entry.track.hits += 1;
            entry.track.misses = 0;

            if entry.track.state == TrackState::Tentative
                && entry.track.hits >= TRACK_PROMOTION_HITS
            {
                entry.track.state = TrackState::Confirmed;
            }
        }

        for (track_idx, entry) in self.entries.iter_mut().enumerate() {
            if matched_entries[track_idx] {
                continue;
            }

            entry.track.misses += 1;
            entry.track.bbox = predicted_boxes[track_idx];
        }

        for (detection_idx, detection) in detections.iter().enumerate() {
            if matched_detections[detection_idx] {
                continue;
            }

            let id = TrackId(self.next_id);
            self.next_id += 1;

            self.entries.push(TrackEntry {
                track: Track {
                    id,
                    class_id: detection.class_id,
                    state: TrackState::Tentative,
                    bbox: detection.bbox,
                    confidence: detection.confidence,
                    first_seen: stamp,
                    last_seen: stamp,
                    hits: 1,
                    misses: 0,
                },
                filter: KalmanBoxTracker::new(&detection.bbox),
            });
        }

        self.entries.retain(|entry| {
            !(entry.track.state == TrackState::Tentative && entry.track.misses > 0)
        });

        self.entries
            .iter()
            .map(|entry| entry.track.clone())
            .collect()
    }

    #[cfg(test)]
    pub fn force_reject_on_next_update(&mut self, id: TrackId) {
        self.test_force_reject_update_for = Some(id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::clock::TimeSource;
    use crate::tracking::params::TRACK_PROMOTION_HITS;

    fn stamp(index: u64, media_ms: f64) -> FrameStamp {
        FrameStamp {
            index,
            media_ms,
            source: TimeSource::Reported,
            adjusted: false,
        }
    }

    fn detection(class_id: u32, cx: f32, cy: f32, w: f32, h: f32) -> Detection {
        Detection {
            class_id,
            confidence: 1.0,
            bbox: BBox::from_center_size(cx, cy, w, h),
        }
    }

    fn single_track_id(tracks: &[Track]) -> TrackId {
        assert_eq!(tracks.len(), 1, "expected exactly one track");
        tracks[0].id
    }

    #[test]
    fn linear_motion_preserves_identity_after_promotion() {
        let mut tracker = Tracker::new();
        let mut expected_id = None;

        for frame in 0..10u64 {
            let cx = 100.0 + frame as f32 * 10.0;
            let tracks = tracker.update(
                &[detection(0, cx, 100.0, 20.0, 20.0)],
                stamp(frame, frame as f64 * 33.0),
            );

            if frame + 1 >= u64::from(TRACK_PROMOTION_HITS) {
                let id = single_track_id(&tracks);
                if let Some(expected_id) = expected_id {
                    assert_eq!(id, expected_id);
                } else {
                    expected_id = Some(id);
                    assert_eq!(id, TrackId(1));
                }
                assert_eq!(tracks[0].state, TrackState::Confirmed);
            }
        }
    }

    #[test]
    fn promotion_requires_three_consecutive_hits() {
        let mut tracker = Tracker::new();

        for frame in 0..2u64 {
            let tracks = tracker.update(
                &[detection(0, 100.0, 100.0, 20.0, 20.0)],
                stamp(frame, frame as f64 * 33.0),
            );
            assert_eq!(tracks.len(), 1);
            assert_eq!(tracks[0].state, TrackState::Tentative);
        }

        let tracks = tracker.update(&[detection(0, 100.0, 100.0, 20.0, 20.0)], stamp(2, 66.0));
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].state, TrackState::Confirmed);
        assert_eq!(tracks[0].hits, TRACK_PROMOTION_HITS);
    }

    #[test]
    fn tentative_track_discarded_on_first_miss_and_id_not_reused() {
        let mut tracker = Tracker::new();

        let tracks = tracker.update(&[detection(0, 100.0, 100.0, 20.0, 20.0)], stamp(0, 0.0));
        assert_eq!(single_track_id(&tracks), TrackId(1));

        let tracks = tracker.update(&[], stamp(1, 33.0));
        assert!(tracks.is_empty());
        assert_eq!(tracker.live_track_count(), 0);

        let tracks = tracker.update(&[detection(0, 100.0, 100.0, 20.0, 20.0)], stamp(2, 66.0));
        assert_eq!(single_track_id(&tracks), TrackId(2));
    }

    #[test]
    fn short_gap_preserves_identity_before_retention_limit() {
        let mut tracker = Tracker::new();

        for frame in 0..TRACK_PROMOTION_HITS as u64 {
            tracker.update(
                &[detection(0, 100.0, 100.0, 20.0, 20.0)],
                stamp(frame, frame as f64 * 33.0),
            );
        }

        tracker.update(&[], stamp(3, 500.0));

        let tracks = tracker.update(&[detection(0, 100.0, 100.0, 20.0, 20.0)], stamp(4, 900.0));
        assert_eq!(single_track_id(&tracks), TrackId(1));
    }

    #[test]
    fn long_gap_issues_new_identity_after_retention_limit() {
        let mut tracker = Tracker::new();

        for frame in 0..TRACK_PROMOTION_HITS as u64 {
            tracker.update(
                &[detection(0, 100.0, 100.0, 20.0, 20.0)],
                stamp(frame, frame as f64 * 33.0),
            );
        }

        tracker.update(&[], stamp(3, 1067.0));

        let tracks = tracker.update(&[detection(0, 100.0, 100.0, 20.0, 20.0)], stamp(4, 1100.0));
        assert_eq!(single_track_id(&tracks), TrackId(2));
    }

    #[test]
    fn same_class_crossing_preserves_both_identities() {
        let mut tracker = Tracker::new();

        for frame in 0..TRACK_PROMOTION_HITS as u64 {
            tracker.update(
                &[
                    detection(0, 100.0, 100.0, 20.0, 20.0),
                    detection(0, 200.0, 100.0, 20.0, 20.0),
                ],
                stamp(frame, frame as f64 * 33.0),
            );
        }

        let left_id = TrackId(1);
        let right_id = TrackId(2);

        tracker.update(
            &[
                detection(0, 130.0, 100.0, 20.0, 20.0),
                detection(0, 170.0, 100.0, 20.0, 20.0),
            ],
            stamp(3, 99.0),
        );

        let tracks = tracker.update(
            &[
                detection(0, 200.0, 100.0, 20.0, 20.0),
                detection(0, 100.0, 100.0, 20.0, 20.0),
            ],
            stamp(4, 132.0),
        );
        assert_eq!(tracks.len(), 2);

        let ids: Vec<TrackId> = tracks.iter().map(|track| track.id).collect();
        assert!(ids.contains(&left_id));
        assert!(ids.contains(&right_id));
    }

    #[test]
    fn rejected_filter_update_keeps_track_and_increments_counter() {
        let mut tracker = Tracker::new();

        for frame in 0..TRACK_PROMOTION_HITS as u64 {
            tracker.update(
                &[detection(0, 100.0, 100.0, 20.0, 20.0)],
                stamp(frame, frame as f64 * 33.0),
            );
        }

        tracker.force_reject_on_next_update(TrackId(1));

        let tracks = tracker.update(&[detection(0, 100.0, 100.0, 20.0, 20.0)], stamp(3, 99.0));
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].id, TrackId(1));
        assert_eq!(tracker.rejected_updates(), 1);
        assert_eq!(tracks[0].hits, TRACK_PROMOTION_HITS + 1);
        assert_eq!(tracks[0].misses, 0);
    }

    #[test]
    fn first_seen_and_last_seen_follow_match_history() {
        let mut tracker = Tracker::new();

        for frame in 0..TRACK_PROMOTION_HITS as u64 {
            tracker.update(
                &[detection(0, 100.0, 100.0, 20.0, 20.0)],
                stamp(frame, frame as f64 * 33.0),
            );
        }

        tracker.update(&[], stamp(3, 500.0));

        let tracks = tracker.update(&[detection(0, 100.0, 100.0, 20.0, 20.0)], stamp(4, 900.0));
        let track = &tracks[0];

        assert_eq!(track.first_seen.index, 0);
        assert_eq!(track.last_seen.index, 4);
    }

    #[test]
    fn association_uses_predicted_boxes_not_last_matched_boxes() {
        let mut tracker = Tracker::new();

        for frame in 0..10u64 {
            let cx = 100.0 + frame as f32 * 10.0;
            tracker.update(
                &[detection(0, cx, 100.0, 20.0, 20.0)],
                stamp(frame, frame as f64 * 33.0),
            );
        }

        let last_matched_cx = 100.0 + 9.0 * 10.0;
        let occluded = tracker.update(&[], stamp(10, 330.0));
        let predicted_cx = occluded[0].bbox.center().0;

        assert!(
            predicted_cx > last_matched_cx + 5.0,
            "predicted center {predicted_cx} should be ahead of last matched {last_matched_cx}"
        );

        let tracks = tracker.update(
            &[detection(0, predicted_cx, 100.0, 20.0, 20.0)],
            stamp(11, 363.0),
        );
        assert_eq!(single_track_id(&tracks), TrackId(1));

        let mut stale_tracker = Tracker::new();
        for frame in 0..10u64 {
            let cx = 100.0 + frame as f32 * 10.0;
            stale_tracker.update(
                &[detection(0, cx, 100.0, 20.0, 20.0)],
                stamp(frame, frame as f64 * 33.0),
            );
        }
        stale_tracker.update(&[], stamp(10, 330.0));

        let stale_tracks = stale_tracker.update(
            &[detection(0, last_matched_cx, 100.0, 20.0, 20.0)],
            stamp(11, 363.0),
        );
        let original = stale_tracks
            .iter()
            .find(|track| track.id == TrackId(1))
            .expect("original track should remain live after a miss");
        assert_eq!(original.misses, 2);
        assert!(
            stale_tracks.iter().any(|track| track.id == TrackId(2)),
            "stale detection should spawn a new identity"
        );
    }

    #[test]
    fn live_track_count_stays_bounded_over_long_sequence() {
        let mut tracker = Tracker::new();
        let mut max_live = 0usize;
        const MAX_CONCURRENT: usize = 3;
        const FRAME_COUNT: u64 = 500;
        const EMPTY_SEGMENT_FRAMES: u64 = 40;

        for frame in 0..FRAME_COUNT {
            let cycle = frame / (MAX_CONCURRENT as u64 * 40 + EMPTY_SEGMENT_FRAMES);
            let phase = frame % (MAX_CONCURRENT as u64 * 40 + EMPTY_SEGMENT_FRAMES);

            let detections = if phase >= MAX_CONCURRENT as u64 * 40 {
                Vec::new()
            } else {
                let slot = (phase / 40) as usize;
                let local = phase % 40;
                vec![detection(
                    0,
                    50.0 + slot as f32 * 80.0 + local as f32,
                    100.0,
                    20.0,
                    20.0,
                )]
            };

            tracker.update(
                &detections,
                stamp(frame + cycle * 1000, frame as f64 * 33.0),
            );
            max_live = max_live.max(tracker.live_track_count());

            if phase >= MAX_CONCURRENT as u64 * 40 + EMPTY_SEGMENT_FRAMES - 1 {
                assert_eq!(
                    tracker.live_track_count(),
                    0,
                    "retention should clear live tracks after an empty segment at frame {frame}"
                );
            }
        }

        assert!(
            max_live <= MAX_CONCURRENT + 2,
            "live tracks peaked at {max_live}, expected bound <= {}",
            MAX_CONCURRENT + 2
        );
    }
}
