use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ndarray::{Array4, ArrayView3};
use opencv::{
    core::{self, BorderTypes, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Outlet, Shape, Tensor, TensorElementType, ValueType};
use ort::{inputs, value::DynValue};

use crate::tracking::BBox;

const INPUT_SIZE: i32 = 640;
const PAD_VALUE: u8 = 114;
const INPUT_SHAPE: [i64; 4] = [1, 3, 640, 640];
const OUTPUT_SHAPE: [i64; 3] = [1, 84, 8400];
const CONFIDENCE_THRESHOLD: f32 = 0.25;
const NMS_IOU_THRESHOLD: f32 = 0.70;
const NUM_CLASSES: usize = 80;
const NUM_PREDICTIONS: usize = 8400;

static COCO_CLASS_NAMES: [&str; NUM_CLASSES] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
];

#[derive(Debug, Clone, PartialEq)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_left: i32,
    pub pad_top: i32,
    pub source_width: i32,
    pub source_height: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Detection {
    pub class_id: u32,
    pub confidence: f32,
    pub bbox: BBox,
}

#[derive(Debug)]
pub struct InferenceResult {
    pub detections: Vec<Detection>,
    pub inference_ms: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct RawCandidate {
    class_id: u32,
    confidence: f32,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
}

pub fn coco_class_name(class_id: u32) -> Option<&'static str> {
    COCO_CLASS_NAMES.get(class_id as usize).copied()
}

pub struct YoloV8Detector {
    session: Session,
    input_name: String,
    output_name: String,
}

impl YoloV8Detector {
    pub fn load(path: &Path) -> Result<Self> {
        let load_start = Instant::now();
        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("failed to create ONNX session builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| anyhow::anyhow!("failed to set ONNX graph optimization level: {e}"))?
            .commit_from_file(path)
            .map_err(|e| {
                anyhow::anyhow!("failed to load ONNX model from {}: {e}", path.display())
            })?;

        let (input_name, output_name) =
            validate_yolov8_contract(session.inputs(), session.outputs())?;

        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
        tracing::info!(
            input_name = %input_name,
            output_name = %output_name,
            input_shape = ?INPUT_SHAPE,
            output_shape = ?OUTPUT_SHAPE,
            load_ms = %format!("{load_ms:.1}"),
            "yolov8 model loaded"
        );

        Ok(Self {
            session,
            input_name,
            output_name,
        })
    }

    pub fn infer(&mut self, frame: &Mat) -> Result<InferenceResult> {
        let (input, transform) = preprocess_frame(frame)?;
        let input_value = Tensor::from_array(input).map_err(ort_error)?;

        let inference_start = Instant::now();
        let outputs = self
            .session
            .run(inputs![self.input_name.as_str() => input_value])
            .map_err(|e| anyhow::anyhow!("ONNX inference failed: {e}"))?;
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

        let output = extract_output_view(&outputs[self.output_name.as_str()])?;
        let detections = postprocess_output(&output, &transform)?;

        Ok(InferenceResult {
            detections,
            inference_ms,
        })
    }
}

pub(crate) fn validate_yolov8_contract(
    inputs: &[Outlet],
    outputs: &[Outlet],
) -> Result<(String, String)> {
    if inputs.len() != 1 {
        bail!(
            "unsupported model: expected exactly 1 input, found {} (expected FP32 tensor shaped [1, 3, 640, 640])",
            inputs.len()
        );
    }
    if outputs.len() != 1 {
        bail!(
            "unsupported model: expected exactly 1 output, found {} (expected FP32 tensor shaped [1, 84, 8400])",
            outputs.len()
        );
    }

    let input = &inputs[0];
    let output = &outputs[0];

    validate_tensor_contract("input", input.name(), input.dtype(), &INPUT_SHAPE)?;
    validate_tensor_contract("output", output.name(), output.dtype(), &OUTPUT_SHAPE)?;

    Ok((input.name().to_string(), output.name().to_string()))
}

fn validate_tensor_contract(
    role: &str,
    name: &str,
    dtype: &ValueType,
    expected_shape: &[i64],
) -> Result<()> {
    let ValueType::Tensor {
        ty,
        shape,
        dimension_symbols: _,
    } = dtype
    else {
        bail!(
            "unsupported model {role} `{name}`: expected FP32 tensor shaped {expected_shape:?}, found {dtype}"
        );
    };

    if *ty != TensorElementType::Float32 {
        bail!(
            "unsupported model {role} `{name}`: expected FP32 tensor shaped {expected_shape:?}, found element type {ty}"
        );
    }

    if !shape_matches(shape, expected_shape) {
        bail!(
            "unsupported model {role} `{name}`: expected FP32 tensor shaped {expected_shape:?}, found shape {shape:?}"
        );
    }

    Ok(())
}

fn shape_matches(shape: &Shape, expected: &[i64]) -> bool {
    if shape.len() != expected.len() {
        return false;
    }

    shape
        .iter()
        .zip(expected.iter())
        .all(|(actual, expected)| *actual == *expected)
}

pub(crate) fn preprocess_frame(frame: &Mat) -> Result<(Array4<f32>, LetterboxTransform)> {
    if frame.empty() {
        bail!("cannot preprocess empty frame");
    }

    let src_w = frame.cols();
    let src_h = frame.rows();
    if src_w <= 0 || src_h <= 0 {
        bail!("cannot preprocess frame with non-positive dimensions: {src_w}x{src_h}");
    }

    if frame.typ() != core::CV_8UC3 {
        bail!(
            "cannot preprocess unsupported frame type {} (expected 8-bit BGR)",
            frame.typ()
        );
    }

    let scale = (INPUT_SIZE as f32 / src_w as f32).min(INPUT_SIZE as f32 / src_h as f32);
    let resized_w = (src_w as f32 * scale).round() as i32;
    let resized_h = (src_h as f32 * scale).round() as i32;
    let pad_left = (INPUT_SIZE - resized_w) / 2;
    let pad_top = (INPUT_SIZE - resized_h) / 2;
    let pad_right = INPUT_SIZE - resized_w - pad_left;
    let pad_bottom = INPUT_SIZE - resized_h - pad_top;

    let mut resized = Mat::default();
    imgproc::resize(
        frame,
        &mut resized,
        Size::new(resized_w, resized_h),
        0.0,
        0.0,
        imgproc::INTER_LINEAR,
    )
    .context("failed to resize frame for letterbox preprocessing")?;

    let mut padded = Mat::default();
    core::copy_make_border(
        &resized,
        &mut padded,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        BorderTypes::BORDER_CONSTANT.into(),
        Scalar::new(
            f64::from(PAD_VALUE),
            f64::from(PAD_VALUE),
            f64::from(PAD_VALUE),
            0.0,
        ),
    )
    .context("failed to pad frame for letterbox preprocessing")?;

    let mut rgb = Mat::default();
    imgproc::cvt_color(&padded, &mut rgb, imgproc::COLOR_BGR2RGB, 0)
        .context("failed to convert frame from BGR to RGB")?;

    // Interleaved HWC bytes are converted to planar NCHW in one contiguous pass;
    // per-pixel `at_2d` lookups dominated preprocessing at ~7.3 ms per 1080p frame.
    if !rgb.is_continuous() {
        bail!("preprocessed RGB frame is not contiguous");
    }

    let plane = (INPUT_SIZE * INPUT_SIZE) as usize;
    let rgb_data = rgb
        .data_bytes()
        .context("failed to access preprocessed RGB frame data")?;
    if rgb_data.len() != plane * 3 {
        bail!(
            "unexpected preprocessed RGB buffer length {} (expected {})",
            rgb_data.len(),
            plane * 3
        );
    }

    let mut tensor = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    {
        let data = tensor
            .as_slice_mut()
            .context("preprocessing tensor is not contiguous")?;
        let (red, rest) = data.split_at_mut(plane);
        let (green, blue) = rest.split_at_mut(plane);

        let (pixels, _) = rgb_data.as_chunks::<3>();
        for (index, pixel) in pixels.iter().enumerate() {
            red[index] = f32::from(pixel[0]) / 255.0;
            green[index] = f32::from(pixel[1]) / 255.0;
            blue[index] = f32::from(pixel[2]) / 255.0;
        }
    }

    let transform = LetterboxTransform {
        scale,
        pad_left,
        pad_top,
        source_width: src_w,
        source_height: src_h,
    };

    Ok((tensor, transform))
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

pub(crate) fn non_maximum_suppression(mut candidates: Vec<Detection>) -> Vec<Detection> {
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

pub(crate) fn postprocess_output(
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
    use ndarray::{Array3, Axis};
    use opencv::core::CV_8UC3;
    use ort::value::SymbolicDimensions;

    fn solid_bgr_mat(width: i32, height: i32, b: u8, g: u8, r: u8) -> Mat {
        Mat::new_rows_cols_with_default(
            height,
            width,
            CV_8UC3,
            Scalar::new(f64::from(b), f64::from(g), f64::from(r), 0.0),
        )
        .expect("failed to create test mat")
    }

    fn tensor_outlet(name: &str, shape: &[i64], ty: TensorElementType) -> Outlet {
        Outlet::new(
            name,
            ValueType::Tensor {
                ty,
                shape: Shape::new(shape.iter().copied()),
                dimension_symbols: SymbolicDimensions::empty(shape.len()),
            },
        )
    }

    #[test]
    fn empty_frame_is_rejected() {
        let frame = Mat::default();
        let err = preprocess_frame(&frame).expect_err("empty frame should fail");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn square_frame_has_unit_scale_and_no_padding() {
        let frame = solid_bgr_mat(640, 640, 0, 0, 255);
        let (tensor, transform) = preprocess_frame(&frame).expect("preprocess should succeed");

        assert!((transform.scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(transform.pad_left, 0);
        assert_eq!(transform.pad_top, 0);
        assert_eq!(transform.source_width, 640);
        assert_eq!(transform.source_height, 640);
        assert_eq!(tensor.shape(), &[1, 3, 640, 640]);
        assert!((tensor[[0, 0, 0, 0]] - 1.0).abs() < f32::EPSILON);
        assert!((tensor[[0, 1, 0, 0]]).abs() < f32::EPSILON);
        assert!((tensor[[0, 2, 0, 0]]).abs() < f32::EPSILON);
    }

    #[test]
    fn landscape_frame_uses_horizontal_letterbox() {
        let frame = solid_bgr_mat(1280, 720, 114, 114, 114);
        let (tensor, transform) = preprocess_frame(&frame).expect("preprocess should succeed");

        assert!((transform.scale - 0.5).abs() < f32::EPSILON);
        assert_eq!(transform.pad_left, 0);
        assert_eq!(transform.pad_top, 140);
        assert_eq!(tensor.shape(), &[1, 3, 640, 640]);

        let pad_value = 114.0_f32 / 255.0;
        assert!((tensor[[0, 0, 0, 0]] - pad_value).abs() < 1e-6);
        assert!((tensor[[0, 1, 0, 0]] - pad_value).abs() < 1e-6);
        assert!((tensor[[0, 2, 0, 0]] - pad_value).abs() < 1e-6);

        let content_y = 140;
        assert!((tensor[[0, 0, content_y, 0]] - pad_value).abs() < 1e-6);
    }

    #[test]
    fn portrait_frame_uses_vertical_letterbox() {
        let frame = solid_bgr_mat(720, 1280, 114, 114, 114);
        let (_, transform) = preprocess_frame(&frame).expect("preprocess should succeed");

        assert!((transform.scale - 0.5).abs() < f32::EPSILON);
        assert_eq!(transform.pad_left, 140);
        assert_eq!(transform.pad_top, 0);
    }

    #[test]
    fn odd_dimensions_assign_padding_remainder_to_right_and_bottom() {
        let frame = solid_bgr_mat(1279, 719, 114, 114, 114);
        let (_, transform) = preprocess_frame(&frame).expect("preprocess should succeed");

        let resized_w = (1279.0_f32 * transform.scale).round() as i32;
        let resized_h = (719.0_f32 * transform.scale).round() as i32;
        let pad_right = INPUT_SIZE - resized_w - transform.pad_left;
        let pad_bottom = INPUT_SIZE - resized_h - transform.pad_top;

        assert_eq!(transform.pad_left, 0);
        assert_eq!(transform.pad_top, 140);
        assert!(pad_right >= 0);
        assert!(pad_bottom >= 0);
        assert_eq!(transform.pad_left + resized_w + pad_right, INPUT_SIZE);
        assert_eq!(transform.pad_top + resized_h + pad_bottom, INPUT_SIZE);
    }

    #[test]
    fn normalization_maps_255_to_one() {
        let frame = solid_bgr_mat(640, 640, 0, 0, 255);
        let (tensor, _) = preprocess_frame(&frame).expect("preprocess should succeed");
        assert!((tensor[[0, 0, 320, 320]] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn contract_rejects_wrong_input_count() {
        let inputs = [
            tensor_outlet("a", &INPUT_SHAPE, TensorElementType::Float32),
            tensor_outlet("b", &INPUT_SHAPE, TensorElementType::Float32),
        ];
        let outputs = [tensor_outlet(
            "out",
            &OUTPUT_SHAPE,
            TensorElementType::Float32,
        )];

        let err = validate_yolov8_contract(&inputs, &outputs).expect_err("should fail");
        assert!(err.to_string().contains("1 input"));
    }

    #[test]
    fn contract_rejects_wrong_output_count() {
        let inputs = [tensor_outlet(
            "images",
            &INPUT_SHAPE,
            TensorElementType::Float32,
        )];
        let outputs = [
            tensor_outlet("out0", &OUTPUT_SHAPE, TensorElementType::Float32),
            tensor_outlet("out1", &OUTPUT_SHAPE, TensorElementType::Float32),
        ];

        let err = validate_yolov8_contract(&inputs, &outputs).expect_err("should fail");
        assert!(err.to_string().contains("1 output"));
    }

    #[test]
    fn contract_rejects_non_fp32_type() {
        let inputs = [tensor_outlet(
            "images",
            &INPUT_SHAPE,
            TensorElementType::Float16,
        )];
        let outputs = [tensor_outlet(
            "out",
            &OUTPUT_SHAPE,
            TensorElementType::Float32,
        )];

        let err = validate_yolov8_contract(&inputs, &outputs).expect_err("should fail");
        assert!(err.to_string().contains("FP32"));
    }

    #[test]
    fn contract_rejects_wrong_input_shape() {
        let inputs = [tensor_outlet(
            "images",
            &[1, 3, 416, 416],
            TensorElementType::Float32,
        )];
        let outputs = [tensor_outlet(
            "out",
            &OUTPUT_SHAPE,
            TensorElementType::Float32,
        )];

        let err = validate_yolov8_contract(&inputs, &outputs).expect_err("should fail");
        assert!(err.to_string().contains("[1, 3, 640, 640]"));
    }

    #[test]
    fn contract_rejects_wrong_output_shape() {
        let inputs = [tensor_outlet(
            "images",
            &INPUT_SHAPE,
            TensorElementType::Float32,
        )];
        let outputs = [tensor_outlet(
            "out",
            &[1, 85, 8400],
            TensorElementType::Float32,
        )];

        let err = validate_yolov8_contract(&inputs, &outputs).expect_err("should fail");
        assert!(err.to_string().contains("[1, 84, 8400]"));
    }

    #[test]
    fn contract_accepts_supported_shapes() {
        let inputs = [tensor_outlet(
            "images",
            &INPUT_SHAPE,
            TensorElementType::Float32,
        )];
        let outputs = [tensor_outlet(
            "output0",
            &OUTPUT_SHAPE,
            TensorElementType::Float32,
        )];

        let (input_name, output_name) =
            validate_yolov8_contract(&inputs, &outputs).expect("contract should pass");
        assert_eq!(input_name, "images");
        assert_eq!(output_name, "output0");
    }

    #[test]
    fn tensor_is_nchw_contiguous() {
        let frame = solid_bgr_mat(640, 640, 10, 20, 30);
        let (tensor, _) = preprocess_frame(&frame).expect("preprocess should succeed");

        let channel_0 = tensor.index_axis(Axis(1), 0);
        let channel_1 = tensor.index_axis(Axis(1), 1);
        let channel_2 = tensor.index_axis(Axis(1), 2);

        assert_eq!(channel_0[[0, 0, 0]], 30.0 / 255.0);
        assert_eq!(channel_1[[0, 0, 0]], 20.0 / 255.0);
        assert_eq!(channel_2[[0, 0, 0]], 10.0 / 255.0);
    }

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
    fn coco_class_name_maps_first_and_last() {
        assert_eq!(coco_class_name(0), Some("person"));
        assert_eq!(coco_class_name(79), Some("toothbrush"));
        assert_eq!(coco_class_name(80), None);
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

    #[test]
    fn load_rejects_invalid_onnx_bytes() {
        let dir = std::env::temp_dir().join(format!(
            "vision-engine-invalid-onnx-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let model_path = dir.join("invalid.onnx");
        std::fs::write(&model_path, b"not an onnx file").expect("failed to write invalid model");

        let err = match YoloV8Detector::load(&model_path) {
            Ok(_) => panic!("invalid onnx should fail"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(message.contains("failed to load ONNX model"));
        assert!(message.contains("invalid.onnx"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
