use std::time::Instant;

use anyhow::{Context, Result, bail};
use ndarray::ArrayView3;
use ort::inputs;
use ort::value::{DynValue, Tensor};
use vision_engine::detector::{Detection, LetterboxTransform, LoadedModel};
use vision_engine::tracking::BBox;

use super::message::{DetectedFrame, PreparedFrame};

const OUTPUT_SHAPE: [i64; 3] = [1, 84, 8400];
const CONFIDENCE_THRESHOLD: f32 = 0.25;
const NMS_IOU_THRESHOLD: f32 = 0.70;
const NUM_CLASSES: usize = 80;
const NUM_PREDICTIONS: usize = 8400;

#[derive(Debug, Clone, PartialEq)]
struct RawCandidate {
    class_id: u32,
    confidence: f32,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

pub struct InferStage {
    model: LoadedModel,
}

impl InferStage {
    pub fn new(model: LoadedModel) -> Self {
        Self { model }
    }

    pub fn detect(&mut self, prepared: PreparedFrame) -> Result<DetectedFrame> {
        let PreparedFrame {
            frame,
            stamp,
            mut timings,
            input,
            transform,
        } = prepared;
        let input_value = Tensor::from_array(input).map_err(ort_error)?;

        let inference_start = Instant::now();
        let outputs = self
            .model
            .session
            .run(inputs![self.model.input_name.as_str() => input_value])
            .map_err(|e| anyhow::anyhow!("ONNX inference failed: {e}"))?;
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

        let output = extract_output_view(&outputs[self.model.output_name.as_str()])?;
        let detections = postprocess_output(&output, &transform)?;
        timings.inference_ms = inference_ms;

        Ok(DetectedFrame {
            frame,
            stamp,
            timings,
            detections,
        })
    }
}

fn extract_output_view(output: &DynValue) -> Result<ArrayView3<'_, f32>> {
    let view = output
        .try_extract_array::<f32>()
        .map_err(ort_error)
        .context("failed to extract model output tensor")?;

    if !output_shape_matches(view.shape()) {
        bail!(
            "unexpected model output shape {:?}, expected {:?}",
            view.shape(),
            OUTPUT_SHAPE
        );
    }

    view.into_dimensionality()
        .context("failed to reshape model output to [1, 84, 8400]")
}

fn output_shape_matches(shape: &[usize]) -> bool {
    shape.len() == OUTPUT_SHAPE.len()
        && shape
            .iter()
            .zip(OUTPUT_SHAPE.iter())
            .all(|(actual, expected)| *actual == *expected as usize)
}

fn ort_error(error: ort::Error) -> anyhow::Error {
    anyhow::Error::new(error)
}

fn is_finite_f32(value: f32) -> bool {
    value.is_finite()
}

fn extract_candidates(output: &ArrayView3<'_, f32>) -> Vec<RawCandidate> {
    let mut candidates = Vec::new();

    for index in 0..NUM_PREDICTIONS {
        let cx = output[[0, 0, index]];
        let cy = output[[0, 1, index]];
        let w = output[[0, 2, index]];
        let h = output[[0, 3, index]];

        if !is_finite_f32(cx)
            || !is_finite_f32(cy)
            || !is_finite_f32(w)
            || !is_finite_f32(h)
            || w <= 0.0
            || h <= 0.0
        {
            continue;
        }

        let (class_id, confidence) = best_class_score(output, index);
        if !is_finite_f32(confidence) || confidence < CONFIDENCE_THRESHOLD {
            continue;
        }

        candidates.push(RawCandidate {
            class_id,
            confidence,
            cx,
            cy,
            w,
            h,
        });
    }

    candidates
}

fn best_class_score(output: &ArrayView3<'_, f32>, index: usize) -> (u32, f32) {
    let mut best_class = 0_u32;
    let mut best_score = output[[0, 4, index]];

    for class in 1..NUM_CLASSES {
        let score = output[[0, 4 + class, index]];
        if score > best_score {
            best_score = score;
            best_class = class as u32;
        }
    }

    (best_class, best_score)
}

fn xywh_to_corners(cx: f32, cy: f32, w: f32, h: f32) -> (f32, f32, f32, f32) {
    let half_w = w / 2.0;
    let half_h = h / 2.0;
    (cx - half_w, cy - half_h, cx + half_w, cy + half_h)
}

fn inverse_letterbox_coordinate(value: f32, pad: i32, scale: f32) -> f32 {
    (value - pad as f32) / scale
}

fn clamp_coordinate(value: f32, max: f32) -> f32 {
    value.clamp(0.0, max)
}

fn restore_to_source(
    candidate: &RawCandidate,
    transform: &LetterboxTransform,
) -> Option<Detection> {
    let (mut x_min, mut y_min, mut x_max, mut y_max) =
        xywh_to_corners(candidate.cx, candidate.cy, candidate.w, candidate.h);

    x_min = inverse_letterbox_coordinate(x_min, transform.pad_left, transform.scale);
    y_min = inverse_letterbox_coordinate(y_min, transform.pad_top, transform.scale);
    x_max = inverse_letterbox_coordinate(x_max, transform.pad_left, transform.scale);
    y_max = inverse_letterbox_coordinate(y_max, transform.pad_top, transform.scale);

    let frame_w = transform.source_width as f32;
    let frame_h = transform.source_height as f32;
    x_min = clamp_coordinate(x_min, frame_w);
    y_min = clamp_coordinate(y_min, frame_h);
    x_max = clamp_coordinate(x_max, frame_w);
    y_max = clamp_coordinate(y_max, frame_h);

    if x_max <= x_min || y_max <= y_min {
        return None;
    }

    Some(Detection {
        class_id: candidate.class_id,
        confidence: candidate.confidence,
        bbox: BBox {
            x_min,
            y_min,
            x_max,
            y_max,
        },
    })
}

fn sort_deterministic(detections: &mut [Detection]) {
    detections.sort_by(|left, right| {
        right
            .confidence
            .total_cmp(&left.confidence)
            .then_with(|| left.class_id.cmp(&right.class_id))
            .then_with(|| left.bbox.x_min.total_cmp(&right.bbox.x_min))
            .then_with(|| left.bbox.y_min.total_cmp(&right.bbox.y_min))
            .then_with(|| left.bbox.x_max.total_cmp(&right.bbox.x_max))
            .then_with(|| left.bbox.y_max.total_cmp(&right.bbox.y_max))
    });
}

fn non_maximum_suppression(mut candidates: Vec<Detection>) -> Vec<Detection> {
    sort_deterministic(&mut candidates);
    let mut kept = Vec::new();
    let mut suppressed = vec![false; candidates.len()];

    for i in 0..candidates.len() {
        if suppressed[i] {
            continue;
        }

        kept.push(candidates[i].clone());

        for j in (i + 1)..candidates.len() {
            if suppressed[j] {
                continue;
            }

            if candidates[i].class_id != candidates[j].class_id {
                continue;
            }

            if candidates[i].bbox.iou(&candidates[j].bbox) > NMS_IOU_THRESHOLD {
                suppressed[j] = true;
            }
        }
    }

    kept
}

fn postprocess_output(
    output: &ArrayView3<'_, f32>,
    transform: &LetterboxTransform,
) -> Result<Vec<Detection>> {
    let raw_candidates = extract_candidates(output);
    let restored = raw_candidates
        .iter()
        .filter_map(|candidate| restore_to_source(candidate, transform))
        .collect::<Vec<Detection>>();

    Ok(non_maximum_suppression(restored))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    fn empty_output_tensor() -> Array3<f32> {
        Array3::<f32>::zeros((1, 84, NUM_PREDICTIONS))
    }

    struct PlantedCandidate {
        index: usize,
        cx: f32,
        cy: f32,
        w: f32,
        h: f32,
        class_id: u32,
        confidence: f32,
    }

    fn plant_candidate(output: &mut Array3<f32>, candidate: PlantedCandidate) {
        output[[0, 0, candidate.index]] = candidate.cx;
        output[[0, 1, candidate.index]] = candidate.cy;
        output[[0, 2, candidate.index]] = candidate.w;
        output[[0, 3, candidate.index]] = candidate.h;
        output[[0, 4 + candidate.class_id as usize, candidate.index]] = candidate.confidence;
    }

    fn square_transform() -> LetterboxTransform {
        LetterboxTransform {
            scale: 1.0,
            pad_left: 0,
            pad_top: 0,
            source_width: 640,
            source_height: 640,
        }
    }

    fn landscape_transform() -> LetterboxTransform {
        LetterboxTransform {
            scale: 0.5,
            pad_left: 0,
            pad_top: 140,
            source_width: 1280,
            source_height: 720,
        }
    }

    #[test]
    fn confidence_below_threshold_is_rejected() {
        let mut output = empty_output_tensor();
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 0,
                cx: 320.0,
                cy: 320.0,
                w: 100.0,
                h: 100.0,
                class_id: 0,
                confidence: 0.24,
            },
        );
        let candidates = extract_candidates(&output.view());
        assert!(candidates.is_empty());
    }

    #[test]
    fn confidence_at_threshold_is_accepted() {
        let mut output = empty_output_tensor();
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 0,
                cx: 320.0,
                cy: 320.0,
                w: 100.0,
                h: 100.0,
                class_id: 0,
                confidence: 0.25,
            },
        );
        let candidates = extract_candidates(&output.view());
        assert_eq!(candidates.len(), 1);
        assert!((candidates[0].confidence - 0.25).abs() < f32::EPSILON);
    }

    #[test]
    fn non_finite_values_are_rejected() {
        let mut output = empty_output_tensor();
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 0,
                cx: f32::NAN,
                cy: 320.0,
                w: 100.0,
                h: 100.0,
                class_id: 0,
                confidence: 0.9,
            },
        );
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 1,
                cx: 320.0,
                cy: f32::INFINITY,
                w: 100.0,
                h: 100.0,
                class_id: 0,
                confidence: 0.9,
            },
        );
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 2,
                cx: 320.0,
                cy: 320.0,
                w: 0.0,
                h: 100.0,
                class_id: 0,
                confidence: 0.9,
            },
        );
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 3,
                cx: 320.0,
                cy: 320.0,
                w: 100.0,
                h: -1.0,
                class_id: 0,
                confidence: 0.9,
            },
        );
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 4,
                cx: 320.0,
                cy: 320.0,
                w: 100.0,
                h: 100.0,
                class_id: 0,
                confidence: f32::NAN,
            },
        );

        let candidates = extract_candidates(&output.view());
        assert!(candidates.is_empty());
    }

    #[test]
    fn xywh_to_corners_computes_expected_box() {
        let (x_min, y_min, x_max, y_max) = xywh_to_corners(320.0, 240.0, 100.0, 80.0);
        assert!((x_min - 270.0).abs() < f32::EPSILON);
        assert!((y_min - 200.0).abs() < f32::EPSILON);
        assert!((x_max - 370.0).abs() < f32::EPSILON);
        assert!((y_max - 280.0).abs() < f32::EPSILON);
    }

    #[test]
    fn restore_square_frame_preserves_coordinates() {
        let candidate = RawCandidate {
            class_id: 0,
            confidence: 0.9,
            cx: 320.0,
            cy: 320.0,
            w: 100.0,
            h: 100.0,
        };
        let detection = restore_to_source(&candidate, &square_transform()).expect("should restore");
        assert!((detection.bbox.x_min - 270.0).abs() < f32::EPSILON);
        assert!((detection.bbox.y_min - 270.0).abs() < f32::EPSILON);
        assert!((detection.bbox.x_max - 370.0).abs() < f32::EPSILON);
        assert!((detection.bbox.y_max - 370.0).abs() < f32::EPSILON);
    }

    #[test]
    fn restore_landscape_frame_reverses_letterbox() {
        let candidate = RawCandidate {
            class_id: 2,
            confidence: 0.8,
            cx: 320.0,
            cy: 360.0,
            w: 80.0,
            h: 60.0,
        };
        let detection =
            restore_to_source(&candidate, &landscape_transform()).expect("should restore");
        assert!((detection.bbox.x_min - 560.0).abs() < 1.0);
        assert!((detection.bbox.y_min - 380.0).abs() < 1.0);
        assert!((detection.bbox.x_max - 720.0).abs() < 1.0);
        assert!((detection.bbox.y_max - 500.0).abs() < 1.0);
    }

    #[test]
    fn restore_clamps_boxes_to_frame_bounds() {
        let candidate = RawCandidate {
            class_id: 0,
            confidence: 0.9,
            cx: 10.0,
            cy: 10.0,
            w: 100.0,
            h: 100.0,
        };
        let detection = restore_to_source(&candidate, &square_transform()).expect("should restore");
        assert!((detection.bbox.x_min - 0.0).abs() < f32::EPSILON);
        assert!((detection.bbox.y_min - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn restore_drops_fully_outside_boxes() {
        let candidate = RawCandidate {
            class_id: 0,
            confidence: 0.9,
            cx: -500.0,
            cy: -500.0,
            w: 10.0,
            h: 10.0,
        };
        assert!(restore_to_source(&candidate, &square_transform()).is_none());
    }

    fn test_bbox(x_min: f32, y_min: f32, x_max: f32, y_max: f32) -> BBox {
        BBox {
            x_min,
            y_min,
            x_max,
            y_max,
        }
    }

    fn test_detection(class_id: u32, confidence: f32, bbox: BBox) -> Detection {
        Detection {
            class_id,
            confidence,
            bbox,
        }
    }

    #[test]
    fn iou_identical_boxes_is_one() {
        let box_a = test_detection(0, 0.9, test_bbox(10.0, 10.0, 50.0, 50.0));
        assert!((box_a.bbox.iou(&box_a.bbox) - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn iou_disjoint_boxes_is_zero() {
        let box_a = test_detection(0, 0.9, test_bbox(0.0, 0.0, 10.0, 10.0));
        let box_b = test_detection(0, 0.8, test_bbox(20.0, 20.0, 30.0, 30.0));
        assert!((box_a.bbox.iou(&box_b.bbox)).abs() < f32::EPSILON);
    }

    #[test]
    fn iou_partial_overlap_is_between_zero_and_one() {
        let box_a = test_detection(0, 0.9, test_bbox(0.0, 0.0, 20.0, 20.0));
        let box_b = test_detection(0, 0.8, test_bbox(10.0, 10.0, 30.0, 30.0));
        let iou = box_a.bbox.iou(&box_b.bbox);
        assert!(iou > 0.0 && iou < 1.0);
    }

    #[test]
    fn nms_suppresses_overlapping_same_class() {
        let high = test_detection(0, 0.9, test_bbox(10.0, 10.0, 50.0, 50.0));
        let low = test_detection(0, 0.8, test_bbox(12.0, 12.0, 48.0, 48.0));
        let kept = non_maximum_suppression(vec![low, high]);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].confidence - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn nms_retains_overlapping_different_classes() {
        let person = test_detection(0, 0.9, test_bbox(10.0, 10.0, 50.0, 50.0));
        let car = test_detection(2, 0.85, test_bbox(12.0, 12.0, 48.0, 48.0));
        let kept = non_maximum_suppression(vec![car, person]);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn deterministic_sort_uses_class_and_coordinates() {
        let a = test_detection(1, 0.5, test_bbox(10.0, 10.0, 20.0, 20.0));
        let b = test_detection(0, 0.5, test_bbox(10.0, 10.0, 20.0, 20.0));
        let mut detections = vec![a, b];
        sort_deterministic(&mut detections);
        assert_eq!(detections[0].class_id, 0);
        assert_eq!(detections[1].class_id, 1);
    }

    #[test]
    fn postprocess_end_to_end_with_synthetic_tensor() {
        let mut output = empty_output_tensor();
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 0,
                cx: 320.0,
                cy: 320.0,
                w: 100.0,
                h: 100.0,
                class_id: 0,
                confidence: 0.9,
            },
        );
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 1,
                cx: 322.0,
                cy: 322.0,
                w: 100.0,
                h: 100.0,
                class_id: 0,
                confidence: 0.8,
            },
        );
        plant_candidate(
            &mut output,
            PlantedCandidate {
                index: 2,
                cx: 320.0,
                cy: 320.0,
                w: 80.0,
                h: 80.0,
                class_id: 2,
                confidence: 0.85,
            },
        );

        let detections = postprocess_output(&output.view(), &square_transform())
            .expect("postprocess should succeed");
        assert_eq!(detections.len(), 2);
        assert!(detections.iter().any(|d| d.class_id == 0));
        assert!(detections.iter().any(|d| d.class_id == 2));
    }
}
