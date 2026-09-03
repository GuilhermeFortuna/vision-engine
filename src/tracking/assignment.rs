use std::collections::BTreeMap;

use crate::detector::Detection;
use crate::tracking::BBox;

/// Cost for padded dummy rows/columns in the assignment matrix. Real IoU costs
/// lie in `[0.0, 1.0]`, so padding stays strictly above any legitimate pair.
const PAD_COST: f32 = 2.0;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Association {
    pub matches: Vec<(usize, usize)>,
    pub unmatched_tracks: Vec<usize>,
    pub unmatched_detections: Vec<usize>,
}

/// `tracks` is `(class_id, predicted box)` in the caller's order.
/// Returned indices address `tracks` and `detections` as given.
pub fn associate(tracks: &[(u32, BBox)], detections: &[Detection], iou_gate: f32) -> Association {
    let mut matches = Vec::new();
    let mut unmatched_tracks = Vec::new();
    let mut unmatched_detections = Vec::new();

    let mut tracks_by_class: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (track_idx, (class_id, _)) in tracks.iter().enumerate() {
        tracks_by_class
            .entry(*class_id)
            .or_default()
            .push(track_idx);
    }

    let mut detections_by_class: BTreeMap<u32, Vec<usize>> = BTreeMap::new();
    for (detection_idx, detection) in detections.iter().enumerate() {
        detections_by_class
            .entry(detection.class_id)
            .or_default()
            .push(detection_idx);
    }

    let class_ids: Vec<u32> = tracks_by_class
        .keys()
        .chain(detections_by_class.keys())
        .copied()
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect();

    for class_id in class_ids {
        let track_indices = tracks_by_class.get(&class_id).cloned().unwrap_or_default();
        let detection_indices = detections_by_class
            .get(&class_id)
            .cloned()
            .unwrap_or_default();

        if track_indices.is_empty() {
            unmatched_detections.extend(detection_indices);
            continue;
        }

        if detection_indices.is_empty() {
            unmatched_tracks.extend(track_indices);
            continue;
        }

        let cost_matrix = build_cost_matrix(
            tracks,
            detections,
            &track_indices,
            &detection_indices,
            iou_gate,
        );
        let local_assignment = solve_assignment(&cost_matrix);
        let mut matched_local_tracks = std::collections::BTreeSet::new();
        let mut matched_local_detections = std::collections::BTreeSet::new();

        for (local_track_idx, local_detection_idx) in &local_assignment {
            matched_local_tracks.insert(*local_track_idx);
            matched_local_detections.insert(*local_detection_idx);

            let track_idx = track_indices[*local_track_idx];
            let detection_idx = detection_indices[*local_detection_idx];
            let track_box = tracks[track_idx].1;
            let detection_box = detections[detection_idx].bbox;
            let iou = track_box.iou(&detection_box);

            // Below-gate edges are already excluded from the cost matrix; re-check
            // here so floating-point edge cases still report unmatched.
            if iou > iou_gate {
                matches.push((track_idx, detection_idx));
            } else {
                unmatched_tracks.push(track_idx);
                unmatched_detections.push(detection_idx);
            }
        }

        for (local_track_idx, &track_idx) in track_indices.iter().enumerate() {
            if !matched_local_tracks.contains(&local_track_idx) {
                unmatched_tracks.push(track_idx);
            }
        }

        for (local_detection_idx, &detection_idx) in detection_indices.iter().enumerate() {
            if !matched_local_detections.contains(&local_detection_idx) {
                unmatched_detections.push(detection_idx);
            }
        }
    }

    matches.sort_by_key(|(track_idx, _)| *track_idx);
    unmatched_tracks.sort_unstable();
    unmatched_detections.sort_unstable();

    Association {
        matches,
        unmatched_tracks,
        unmatched_detections,
    }
}

fn association_cost(track_box: &BBox, detection_box: &BBox, iou_gate: f32) -> f32 {
    let iou = track_box.iou(detection_box);
    if iou.is_finite() && iou > iou_gate {
        1.0 - iou
    } else {
        // Keep invalid pairs strictly above any in-gate cost so Hungarian cannot
        // prefer two below-gate matches over one valid match.
        PAD_COST
    }
}

fn build_cost_matrix(
    tracks: &[(u32, BBox)],
    detections: &[Detection],
    track_indices: &[usize],
    detection_indices: &[usize],
    iou_gate: f32,
) -> Vec<Vec<f32>> {
    track_indices
        .iter()
        .map(|&track_idx| {
            detection_indices
                .iter()
                .map(|&detection_idx| {
                    association_cost(
                        &tracks[track_idx].1,
                        &detections[detection_idx].bbox,
                        iou_gate,
                    )
                })
                .collect()
        })
        .collect()
}

/// Minimum-cost one-to-one assignment between rows and columns.
/// Returns `(local_row_idx, local_col_idx)` pairs. Unassigned rows/columns are omitted.
fn solve_assignment(cost: &[Vec<f32>]) -> Vec<(usize, usize)> {
    let n_rows = cost.len();
    if n_rows == 0 {
        return Vec::new();
    }

    let n_cols = cost[0].len();
    if n_cols == 0 {
        return Vec::new();
    }

    let n = n_rows.max(n_cols);
    let mut square = vec![vec![PAD_COST; n]; n];
    for (row_idx, row) in cost.iter().enumerate().take(n_rows) {
        for (col_idx, &value) in row.iter().enumerate().take(n_cols) {
            square[row_idx][col_idx] = value;
        }
    }

    let column_for_row = hungarian(&square);

    let mut pairs = Vec::new();
    for (row_idx, col_idx) in column_for_row.into_iter().enumerate().take(n_rows) {
        if col_idx < n_cols {
            pairs.push((row_idx, col_idx));
        }
    }

    pairs
}

/// Kuhn-Munkres on a square cost matrix. Returns, for each row, the assigned column.
fn hungarian(cost: &[Vec<f32>]) -> Vec<usize> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }

    let mut u = vec![0.0; n + 1];
    let mut v = vec![0.0; n + 1];
    let mut column_for_row = vec![0usize; n + 1];
    let mut predecessor = vec![0usize; n + 1];

    for row in 1..=n {
        column_for_row[0] = row;
        let mut current_col = 0usize;
        let mut min_reduced = vec![LARGE; n + 1];
        let mut used = vec![false; n + 1];

        loop {
            used[current_col] = true;
            let current_row = column_for_row[current_col];
            let mut delta = LARGE;
            let mut next_col = 0usize;

            for col in 1..=n {
                if used[col] {
                    continue;
                }

                let reduced = cost[current_row - 1][col - 1] - u[current_row] - v[col];
                if reduced < min_reduced[col] {
                    min_reduced[col] = reduced;
                    predecessor[col] = current_col;
                }
                if min_reduced[col] < delta {
                    delta = min_reduced[col];
                    next_col = col;
                }
            }

            for col in 0..=n {
                if used[col] {
                    u[column_for_row[col]] += delta;
                    v[col] -= delta;
                } else {
                    min_reduced[col] -= delta;
                }
            }

            current_col = next_col;
            if column_for_row[current_col] == 0 {
                break;
            }
        }

        loop {
            let prev_col = predecessor[current_col];
            column_for_row[current_col] = column_for_row[prev_col];
            current_col = prev_col;
            if current_col == 0 {
                break;
            }
        }
    }

    let mut assignment = vec![0usize; n];
    for (col, row) in column_for_row.iter().enumerate().skip(1) {
        if *row != 0 {
            assignment[row - 1] = col - 1;
        }
    }

    assignment
}

const LARGE: f32 = 1.0e9;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::detector::Detection;

    fn greedy_row_assignment(cost: &[Vec<f32>]) -> Vec<(usize, usize)> {
        let n_rows = cost.len();
        let n_cols = cost[0].len();
        let mut used_cols = vec![false; n_cols];
        let mut pairs = Vec::new();

        for (row_idx, row) in cost.iter().enumerate().take(n_rows) {
            let mut best_col = None;
            let mut best_cost = f32::MAX;
            for (col_idx, &value) in row.iter().enumerate().take(n_cols) {
                if used_cols[col_idx] {
                    continue;
                }
                if value < best_cost {
                    best_cost = value;
                    best_col = Some(col_idx);
                }
            }
            if let Some(col_idx) = best_col {
                used_cols[col_idx] = true;
                pairs.push((row_idx, col_idx));
            }
        }

        pairs
    }

    fn assignment_cost(pairs: &[(usize, usize)], cost: &[Vec<f32>]) -> f32 {
        pairs
            .iter()
            .map(|(row_idx, col_idx)| cost[*row_idx][*col_idx])
            .sum()
    }

    fn assert_total_coverage(track_count: usize, detection_count: usize, result: &Association) {
        let mut seen_tracks = vec![false; track_count];
        let mut seen_detections = vec![false; detection_count];

        for (track_idx, detection_idx) in &result.matches {
            assert!(*track_idx < track_count);
            assert!(*detection_idx < detection_count);
            assert!(!seen_tracks[*track_idx]);
            assert!(!seen_detections[*detection_idx]);
            seen_tracks[*track_idx] = true;
            seen_detections[*detection_idx] = true;
        }

        for track_idx in &result.unmatched_tracks {
            assert!(*track_idx < track_count);
            assert!(!seen_tracks[*track_idx]);
            seen_tracks[*track_idx] = true;
        }

        for detection_idx in &result.unmatched_detections {
            assert!(*detection_idx < detection_count);
            assert!(!seen_detections[*detection_idx]);
            seen_detections[*detection_idx] = true;
        }

        assert!(seen_tracks.iter().all(|seen| *seen));
        assert!(seen_detections.iter().all(|seen| *seen));
    }

    fn detection(class_id: u32, cx: f32, cy: f32, w: f32, h: f32) -> Detection {
        Detection {
            class_id,
            confidence: 1.0,
            bbox: BBox::from_center_size(cx, cy, w, h),
        }
    }

    fn bbox_xyxy(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> BBox {
        BBox {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    fn detection_xyxy(class_id: u32, x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> Detection {
        Detection {
            class_id,
            confidence: 1.0,
            bbox: bbox_xyxy(x_min, y_min, x_max, y_max),
        }
    }

    #[test]
    fn solver_beats_greedy_row_assignment_on_three_by_three() {
        let cost = vec![
            vec![9.0, 2.0, 7.0],
            vec![6.0, 4.0, 3.0],
            vec![5.0, 8.0, 1.0],
        ];

        let greedy = greedy_row_assignment(&cost);
        let optimal = solve_assignment(&cost);

        assert_eq!(greedy, vec![(0, 1), (1, 2), (2, 0)]);
        assert_eq!(optimal, vec![(0, 1), (1, 0), (2, 2)]);
        assert!(
            assignment_cost(&optimal, &cost) < assignment_cost(&greedy, &cost),
            "solver must beat greedy row assignment"
        );
    }

    #[test]
    fn solver_handles_more_tracks_than_detections() {
        let cost = vec![vec![0.2, 0.8], vec![0.9, 0.1], vec![0.4, 0.6]];

        assert_eq!(solve_assignment(&cost), vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn solver_handles_more_detections_than_tracks() {
        let cost = vec![vec![0.2, 0.8, 0.5], vec![0.9, 0.1, 0.7]];

        assert_eq!(solve_assignment(&cost), vec![(0, 0), (1, 1)]);
    }

    #[test]
    fn solver_handles_empty_inputs() {
        assert!(solve_assignment(&[]).is_empty());
        assert!(solve_assignment(&[vec![]]).is_empty());
    }

    #[test]
    fn solver_breaks_equal_cost_ties_toward_lowest_index() {
        let cost = vec![vec![1.0, 1.0], vec![1.0, 1.0]];

        let first = solve_assignment(&cost);
        let second = solve_assignment(&cost);

        assert_eq!(first, vec![(0, 0), (1, 1)]);
        assert_eq!(first, second);
    }

    #[test]
    fn associate_matches_clean_one_to_one_correspondence() {
        let tracks = vec![
            (0, BBox::from_center_size(10.0, 10.0, 20.0, 20.0)),
            (0, BBox::from_center_size(50.0, 10.0, 20.0, 20.0)),
            (0, BBox::from_center_size(90.0, 10.0, 20.0, 20.0)),
        ];
        let detections = vec![
            detection(0, 10.0, 10.0, 20.0, 20.0),
            detection(0, 50.0, 10.0, 20.0, 20.0),
            detection(0, 90.0, 10.0, 20.0, 20.0),
        ];

        let result = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert_eq!(result.matches, vec![(0, 0), (1, 1), (2, 2)]);
        assert!(result.unmatched_tracks.is_empty());
        assert!(result.unmatched_detections.is_empty());
    }

    #[test]
    fn associate_reports_surplus_detections() {
        let tracks = vec![(0, BBox::from_center_size(10.0, 10.0, 20.0, 20.0))];
        let detections = vec![
            detection(0, 10.0, 10.0, 20.0, 20.0),
            detection(0, 50.0, 10.0, 20.0, 20.0),
        ];

        let result = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert_eq!(result.matches, vec![(0, 0)]);
        assert!(result.unmatched_tracks.is_empty());
        assert_eq!(result.unmatched_detections, vec![1]);
    }

    #[test]
    fn associate_reports_surplus_tracks() {
        let tracks = vec![
            (0, BBox::from_center_size(10.0, 10.0, 20.0, 20.0)),
            (0, BBox::from_center_size(50.0, 10.0, 20.0, 20.0)),
        ];
        let detections = vec![detection(0, 10.0, 10.0, 20.0, 20.0)];

        let result = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert_eq!(result.matches, vec![(0, 0)]);
        assert_eq!(result.unmatched_tracks, vec![1]);
        assert!(result.unmatched_detections.is_empty());
    }

    #[test]
    fn associate_handles_empty_detections() {
        let tracks = vec![(0, BBox::from_center_size(10.0, 10.0, 20.0, 20.0))];
        let detections = Vec::<Detection>::new();

        let result = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_tracks, vec![0]);
        assert!(result.unmatched_detections.is_empty());
    }

    #[test]
    fn associate_handles_empty_tracks() {
        let tracks = Vec::<(u32, BBox)>::new();
        let detections = vec![detection(0, 10.0, 10.0, 20.0, 20.0)];

        let result = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert!(result.matches.is_empty());
        assert!(result.unmatched_tracks.is_empty());
        assert_eq!(result.unmatched_detections, vec![0]);
    }

    #[test]
    fn associate_never_matches_across_classes() {
        let tracks = vec![(0, BBox::from_center_size(10.0, 10.0, 20.0, 20.0))];
        let detections = vec![detection(2, 10.0, 10.0, 20.0, 20.0)];

        let result = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_tracks, vec![0]);
        assert_eq!(result.unmatched_detections, vec![0]);
    }

    #[test]
    fn associate_solves_each_class_independently() {
        let tracks = vec![
            (0, BBox::from_center_size(10.0, 10.0, 20.0, 20.0)),
            (2, BBox::from_center_size(50.0, 10.0, 20.0, 20.0)),
        ];
        let detections = vec![
            detection(0, 10.0, 10.0, 20.0, 20.0),
            detection(2, 50.0, 10.0, 20.0, 20.0),
        ];

        let result = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert_eq!(result.matches, vec![(0, 0), (1, 1)]);
        assert!(result.unmatched_tracks.is_empty());
        assert!(result.unmatched_detections.is_empty());
    }

    #[test]
    fn associate_rejects_pairs_below_gate_after_assignment() {
        let tracks = vec![
            (0, BBox::from_center_size(10.0, 10.0, 20.0, 20.0)),
            (0, BBox::from_center_size(50.0, 10.0, 20.0, 20.0)),
        ];
        let detections = vec![
            detection(0, 25.0, 10.0, 20.0, 20.0),
            detection(0, 65.0, 10.0, 20.0, 20.0),
        ];

        let iou_track0 = tracks[0].1.iou(&detections[0].bbox);
        let iou_track1 = tracks[1].1.iou(&detections[1].bbox);
        assert!(iou_track0 > 0.0 && iou_track0 <= 0.30);
        assert!(iou_track1 > 0.0 && iou_track1 <= 0.30);

        let result = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert!(result.matches.is_empty());
        assert_eq!(result.unmatched_tracks, vec![0, 1]);
        assert_eq!(result.unmatched_detections, vec![0, 1]);
    }

    #[test]
    fn associate_prefers_one_in_gate_match_over_two_below_gate() {
        // Without pre-gating, Hungarian can pick two below-gate pairs whose
        // combined (1 - IoU) beats one valid match plus a poor leftover.
        let tracks = vec![
            (0, bbox_xyxy(0.0, 0.0, 20.0, 20.0)),
            (0, bbox_xyxy(9.3, 0.0, 29.3, 20.0)),
        ];
        let detections = vec![
            detection_xyxy(0, 2.6, 0.0, 22.6, 20.0),
            detection_xyxy(0, 0.0, 6.7, 20.0, 26.7),
        ];

        let gate = 0.50;
        let iou_00 = tracks[0].1.iou(&detections[0].bbox);
        let iou_01 = tracks[0].1.iou(&detections[1].bbox);
        let iou_10 = tracks[1].1.iou(&detections[0].bbox);
        let iou_11 = tracks[1].1.iou(&detections[1].bbox);

        assert!(iou_00 > gate, "T0-D0 must be in-gate ({iou_00})");
        assert!(iou_01 <= gate, "T0-D1 must be below gate ({iou_01})");
        assert!(iou_10 <= gate, "T1-D0 must be below gate ({iou_10})");
        assert!(iou_11 <= gate, "T1-D1 must be below gate ({iou_11})");

        let ungated_cross = (1.0 - iou_01) + (1.0 - iou_10);
        let ungated_diag = (1.0 - iou_00) + (1.0 - iou_11);
        assert!(
            ungated_cross < ungated_diag,
            "setup must make the two below-gate pairs cheaper without pre-gating \
             (cross={ungated_cross}, diag={ungated_diag})"
        );

        let result = associate(&tracks, &detections, gate);
        assert_total_coverage(tracks.len(), detections.len(), &result);
        assert_eq!(result.matches, vec![(0, 0)]);
        assert_eq!(result.unmatched_tracks, vec![1]);
        assert_eq!(result.unmatched_detections, vec![1]);
    }

    #[test]
    fn associate_is_deterministic_across_repeated_calls() {
        let tracks = vec![
            (0, BBox::from_center_size(10.0, 10.0, 20.0, 20.0)),
            (0, BBox::from_center_size(30.0, 10.0, 20.0, 20.0)),
            (0, BBox::from_center_size(50.0, 10.0, 20.0, 20.0)),
        ];
        let detections = vec![
            detection(0, 10.0, 10.0, 20.0, 20.0),
            detection(0, 30.0, 10.0, 20.0, 20.0),
            detection(0, 50.0, 10.0, 20.0, 20.0),
        ];

        let baseline = associate(&tracks, &detections, 0.30);
        assert_total_coverage(tracks.len(), detections.len(), &baseline);

        for _ in 0..10 {
            let repeated = associate(&tracks, &detections, 0.30);
            assert_eq!(repeated, baseline);
        }
    }
}
