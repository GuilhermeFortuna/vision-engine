use vision_engine::detector::Detection;
use vision_engine::tracking::{
    BBox, FrameClock, FrameStamp, TimeSource, Track, TrackState, Tracker,
};

const FPS: f64 = 30.0;

fn run_sequence(frames: &[Vec<Detection>], fps: f64) -> Vec<Vec<Track>> {
    let mut tracker = Tracker::new();
    frames
        .iter()
        .enumerate()
        .map(|(index, detections)| {
            tracker
                .try_update(
                    detections,
                    FrameStamp {
                        index: index as u64,
                        media_ms: index as f64 / fps * 1000.0,
                        source: TimeSource::Reported,
                        adjusted: false,
                    },
                )
                .expect("synthetic tracker sequence should be valid")
        })
        .collect()
}

fn ids_of(tracks: &[Track]) -> Vec<u64> {
    let mut ids: Vec<u64> = tracks
        .iter()
        .filter(|track| track.state == TrackState::Confirmed)
        .map(|track| track.id.0)
        .collect();
    ids.sort_unstable();
    ids
}

fn moving_box(frame: usize, class_id: u32) -> Detection {
    Detection {
        class_id,
        confidence: 0.9,
        bbox: BBox::from_center_size(100.0 + frame as f32 * 2.0, 100.0, 20.0, 20.0),
    }
}

fn stamp(index: u64, media_ms: f64) -> FrameStamp {
    FrameStamp {
        index,
        media_ms,
        source: TimeSource::Reported,
        adjusted: false,
    }
}

#[test]
fn continuous_motion_preserves_the_exact_confirmed_identity_on_every_frame() {
    let frames: Vec<Vec<Detection>> = (0..20).map(|frame| vec![moving_box(frame, 0)]).collect();
    let results = run_sequence(&frames, FPS);
    let identities: Vec<Vec<u64>> = results.iter().map(|tracks| ids_of(tracks)).collect();

    assert_eq!(identities[0], Vec::<u64>::new());
    assert_eq!(identities[1], Vec::<u64>::new());
    for (frame, identities) in identities.iter().enumerate().skip(2) {
        assert_eq!(identities, &vec![1], "frame {frame}");
    }
}

#[test]
fn short_occlusion_gap_preserves_the_exact_identity() {
    let mut tracker = Tracker::new();
    let mut initial_identities = Vec::new();
    for frame in 0..3 {
        let tracks = tracker
            .try_update(
                &[moving_box(frame, 0)],
                stamp(frame as u64, frame as f64 * 33.0),
            )
            .expect("initial detection should succeed");
        initial_identities.push(ids_of(&tracks));
    }
    let occluded = tracker.update(&[], stamp(3, 99.0));
    let tracks = tracker
        .try_update(&[moving_box(4, 0)], stamp(4, 132.0))
        .expect("reappearance should succeed");

    assert_eq!(initial_identities, vec![vec![], vec![], vec![1]]);
    assert_eq!(ids_of(&occluded), vec![1]);
    assert_eq!(ids_of(&tracks), vec![1]);
}

#[test]
fn long_occlusion_gap_issues_a_new_exact_identity() {
    let mut tracker = Tracker::new();
    let mut initial_identities = Vec::new();
    for frame in 0..3 {
        let tracks = tracker
            .try_update(
                &[moving_box(frame, 0)],
                stamp(frame as u64, frame as f64 * 33.0),
            )
            .expect("initial detection should succeed");
        initial_identities.push(ids_of(&tracks));
    }
    let expired = tracker.update(&[], stamp(3, 1_100.0));

    let first = tracker
        .try_update(&[moving_box(4, 0)], stamp(4, 1_133.0))
        .expect("new detection should succeed");
    let second = tracker
        .try_update(&[moving_box(5, 0)], stamp(5, 1_166.0))
        .expect("new detection should succeed");
    let third = tracker
        .try_update(&[moving_box(6, 0)], stamp(6, 1_199.0))
        .expect("new detection should succeed");

    assert_eq!(initial_identities, vec![vec![], vec![], vec![1]]);
    assert_eq!(ids_of(&expired), Vec::<u64>::new());
    assert_eq!(ids_of(&first), Vec::<u64>::new());
    assert_eq!(ids_of(&second), Vec::<u64>::new());
    assert_eq!(ids_of(&third), vec![2]);
}

#[test]
fn different_classes_can_overlap_without_identity_interference() {
    let frames: Vec<Vec<Detection>> = (0..5)
        .map(|frame| vec![moving_box(frame, 0), moving_box(frame, 1)])
        .collect();
    let results = run_sequence(&frames, FPS);

    assert_eq!(ids_of(&results[0]), Vec::<u64>::new());
    assert_eq!(ids_of(&results[1]), Vec::<u64>::new());
    for (frame, tracks) in results.iter().enumerate().skip(2) {
        assert_eq!(ids_of(tracks), vec![1, 2], "frame {frame}");
    }
}

#[test]
fn one_frame_spurious_detection_never_receives_a_confirmed_identity() {
    let results = run_sequence(&[vec![moving_box(0, 0)], vec![], vec![], vec![]], FPS);
    for (frame, tracks) in results.iter().enumerate() {
        assert_eq!(ids_of(tracks), Vec::<u64>::new(), "frame {frame}");
    }
}

#[test]
fn same_class_crossing_records_the_identity_outcome_without_gating_it() {
    let mut frames = Vec::new();
    for (left, right) in [(100.0, 180.0), (110.0, 170.0), (120.0, 160.0)] {
        frames.push(vec![
            Detection {
                class_id: 0,
                confidence: 0.9,
                bbox: BBox::from_center_size(left, 100.0, 20.0, 20.0),
            },
            Detection {
                class_id: 0,
                confidence: 0.9,
                bbox: BBox::from_center_size(right, 100.0, 20.0, 20.0),
            },
        ]);
    }
    for (left, right) in [
        (135.0, 145.0),
        (140.0, 140.0),
        (145.0, 135.0),
        (160.0, 120.0),
    ] {
        frames.push(vec![
            Detection {
                class_id: 0,
                confidence: 0.9,
                bbox: BBox::from_center_size(left, 100.0, 20.0, 20.0),
            },
            Detection {
                class_id: 0,
                confidence: 0.9,
                bbox: BBox::from_center_size(right, 100.0, 20.0, 20.0),
            },
        ]);
    }

    let results = run_sequence(&frames, FPS);
    let before = ids_of(&results[2]);
    let after = ids_of(results.last().expect("crossing result"));
    assert_eq!(before.len(), 2);
    assert_eq!(after.len(), 2);
    println!(
        "same-class crossing ids before={before:?} after={after:?} survived={}",
        before == after
    );
}

#[test]
fn normal_tracker_conditions_continue_without_error() {
    let mut tracker = Tracker::new();
    assert!(tracker.try_update(&[], stamp(0, 0.0)).is_ok());
    assert_eq!(tracker.live_track_count(), 0);

    let mut clock = FrameClock::new(None);
    let fallback_stamp = clock.stamp(None);
    assert_eq!(fallback_stamp.source, TimeSource::DerivedFromIndex);
    assert!(tracker.try_update(&[], fallback_stamp).is_ok());
}

#[test]
fn invalid_tracking_inputs_name_the_stage_and_frame_index() {
    let mut tracker = Tracker::new();
    let timestamp_error = tracker
        .try_update(&[], stamp(42, f64::NAN))
        .expect_err("invalid timestamp must fail");
    assert!(timestamp_error.to_string().contains("timestamp"));
    assert!(timestamp_error.to_string().contains("42"));

    let invalid_detection = Detection {
        class_id: 0,
        confidence: 0.9,
        bbox: BBox::from_center_size(0.0, 0.0, 0.0, 1.0),
    };
    let association_error = tracker
        .try_update(&[invalid_detection], stamp(43, 0.0))
        .expect_err("invalid detection must fail");
    assert!(association_error.to_string().contains("association"));
    assert!(association_error.to_string().contains("43"));
}
