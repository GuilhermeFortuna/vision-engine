use std::path::Path;
use std::time::Instant;

use anyhow::{Context, Result, bail};
use ndarray::{Array3, Array4};
use opencv::{
    core::{self, BorderTypes, Mat, Scalar, Size, Vec3b},
    imgproc,
    prelude::*,
};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Outlet, Shape, Tensor, TensorElementType, ValueType};
use ort::{inputs, value::DynValue};

const INPUT_SIZE: i32 = 640;
const PAD_VALUE: u8 = 114;
const INPUT_SHAPE: [i64; 4] = [1, 3, 640, 640];
const OUTPUT_SHAPE: [i64; 3] = [1, 84, 8400];

#[derive(Debug, Clone, PartialEq)]
pub struct LetterboxTransform {
    pub scale: f32,
    pub pad_left: i32,
    pub pad_top: i32,
    pub source_width: i32,
    pub source_height: i32,
}

#[derive(Debug)]
pub struct RawInferenceResult {
    pub output: Array3<f32>,
    pub transform: LetterboxTransform,
    pub inference_ms: f64,
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

    pub fn infer(&mut self, frame: &Mat) -> Result<RawInferenceResult> {
        let (input, transform) = preprocess_frame(frame)?;
        let input_value = Tensor::from_array(input).map_err(ort_error)?;

        let inference_start = Instant::now();
        let outputs = self
            .session
            .run(inputs![self.input_name.as_str() => input_value])
            .map_err(|e| anyhow::anyhow!("ONNX inference failed: {e}"))?;
        let inference_ms = inference_start.elapsed().as_secs_f64() * 1000.0;

        let output = extract_output_tensor(&outputs[self.output_name.as_str()])?;

        Ok(RawInferenceResult {
            output,
            transform,
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

    let mut tensor = Array4::<f32>::zeros((1, 3, INPUT_SIZE as usize, INPUT_SIZE as usize));
    for y in 0..INPUT_SIZE {
        for x in 0..INPUT_SIZE {
            let pixel = rgb
                .at_2d::<Vec3b>(y, x)
                .context("failed to read preprocessed RGB pixel")?;
            for channel in 0..3 {
                tensor[[0, channel, y as usize, x as usize]] = f32::from(pixel[channel]) / 255.0;
            }
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

fn extract_output_tensor(output: &DynValue) -> Result<Array3<f32>> {
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

    view.to_owned()
        .into_dimensionality()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Axis;
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
}
