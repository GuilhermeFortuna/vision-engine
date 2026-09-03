use std::path::Path;
use std::time::Instant;

use anyhow::{Result, bail};
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;
use ort::value::{Outlet, Shape, TensorElementType, ValueType};

use crate::tracking::BBox;

const INPUT_SHAPE: [i64; 4] = [1, 3, 640, 640];
const OUTPUT_SHAPE: [i64; 3] = [1, 84, 8400];
const NUM_CLASSES: usize = 80;

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

pub struct LoadedModel {
    pub session: Session,
    pub input_name: String,
    pub output_name: String,
}

impl LoadedModel {
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
}

pub fn coco_class_name(class_id: u32) -> Option<&'static str> {
    COCO_CLASS_NAMES.get(class_id as usize).copied()
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

#[cfg(test)]
mod tests {
    use super::*;
    use ort::value::SymbolicDimensions;

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
    fn coco_class_name_maps_first_and_last() {
        assert_eq!(coco_class_name(0), Some("person"));
        assert_eq!(coco_class_name(79), Some("toothbrush"));
        assert_eq!(coco_class_name(80), None);
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

        let err = match LoadedModel::load(&model_path) {
            Ok(_) => panic!("invalid onnx should fail"),
            Err(err) => err,
        };
        let message = err.to_string();
        assert!(message.contains("failed to load ONNX model"));
        assert!(message.contains("invalid.onnx"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
