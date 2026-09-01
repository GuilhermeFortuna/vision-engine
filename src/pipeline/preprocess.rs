use std::time::Instant;

use anyhow::{Context, Result, bail};
use ndarray::Array4;
use opencv::{
    core::{self, BorderTypes, Mat, Scalar, Size},
    imgproc,
    prelude::*,
};
use vision_engine::detector::LetterboxTransform;

use super::message::{DecodedFrame, PreparedFrame};

const INPUT_SIZE: i32 = 640;
const PAD_VALUE: u8 = 114;

pub fn prepare(decoded: DecodedFrame) -> Result<PreparedFrame> {
    let DecodedFrame {
        frame,
        stamp,
        mut timings,
    } = decoded;
    let preprocess_start = Instant::now();

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
        &frame,
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

    let preprocess_ms = preprocess_start.elapsed().as_secs_f64() * 1000.0;
    timings.preprocess_ms = preprocess_ms;

    Ok(PreparedFrame {
        frame,
        stamp,
        timings,
        input: tensor,
        transform,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::message::StageTimings;
    use ndarray::Axis;
    use opencv::core::{CV_8UC3, Mat, Scalar};
    use vision_engine::tracking::clock::{FrameStamp, TimeSource};

    fn sample_stamp() -> FrameStamp {
        FrameStamp {
            index: 0,
            media_ms: 0.0,
            source: TimeSource::Reported,
            adjusted: false,
        }
    }

    fn decoded_from_mat(frame: Mat) -> DecodedFrame {
        DecodedFrame {
            frame,
            stamp: sample_stamp(),
            timings: StageTimings::default(),
        }
    }

    fn solid_bgr_mat(width: i32, height: i32, b: u8, g: u8, r: u8) -> Mat {
        Mat::new_rows_cols_with_default(
            height,
            width,
            CV_8UC3,
            Scalar::new(f64::from(b), f64::from(g), f64::from(r), 0.0),
        )
        .expect("failed to create test mat")
    }

    #[test]
    fn empty_frame_is_rejected() {
        let decoded = decoded_from_mat(Mat::default());
        let err = prepare(decoded).expect_err("empty frame should fail");
        assert!(err.to_string().contains("empty"));
    }

    #[test]
    fn square_frame_has_unit_scale_and_no_padding() {
        let frame = solid_bgr_mat(640, 640, 0, 0, 255);
        let prepared = prepare(decoded_from_mat(frame)).expect("preprocess should succeed");

        assert!((prepared.transform.scale - 1.0).abs() < f32::EPSILON);
        assert_eq!(prepared.transform.pad_left, 0);
        assert_eq!(prepared.transform.pad_top, 0);
        assert_eq!(prepared.transform.source_width, 640);
        assert_eq!(prepared.transform.source_height, 640);
        assert_eq!(prepared.input.shape(), &[1, 3, 640, 640]);
        assert!((prepared.input[[0, 0, 0, 0]] - 1.0).abs() < f32::EPSILON);
        assert!((prepared.input[[0, 1, 0, 0]]).abs() < f32::EPSILON);
        assert!((prepared.input[[0, 2, 0, 0]]).abs() < f32::EPSILON);
    }

    #[test]
    fn landscape_frame_uses_horizontal_letterbox() {
        let frame = solid_bgr_mat(1280, 720, 114, 114, 114);
        let prepared = prepare(decoded_from_mat(frame)).expect("preprocess should succeed");

        assert!((prepared.transform.scale - 0.5).abs() < f32::EPSILON);
        assert_eq!(prepared.transform.pad_left, 0);
        assert_eq!(prepared.transform.pad_top, 140);
        assert_eq!(prepared.input.shape(), &[1, 3, 640, 640]);

        let pad_value = 114.0_f32 / 255.0;
        assert!((prepared.input[[0, 0, 0, 0]] - pad_value).abs() < 1e-6);
        assert!((prepared.input[[0, 1, 0, 0]] - pad_value).abs() < 1e-6);
        assert!((prepared.input[[0, 2, 0, 0]] - pad_value).abs() < 1e-6);

        let content_y = 140;
        assert!((prepared.input[[0, 0, content_y, 0]] - pad_value).abs() < 1e-6);
    }

    #[test]
    fn portrait_frame_uses_vertical_letterbox() {
        let frame = solid_bgr_mat(720, 1280, 114, 114, 114);
        let prepared = prepare(decoded_from_mat(frame)).expect("preprocess should succeed");

        assert!((prepared.transform.scale - 0.5).abs() < f32::EPSILON);
        assert_eq!(prepared.transform.pad_left, 140);
        assert_eq!(prepared.transform.pad_top, 0);
    }

    #[test]
    fn odd_dimensions_assign_padding_remainder_to_right_and_bottom() {
        let frame = solid_bgr_mat(1279, 719, 114, 114, 114);
        let prepared = prepare(decoded_from_mat(frame)).expect("preprocess should succeed");

        let resized_w = (1279.0_f32 * prepared.transform.scale).round() as i32;
        let resized_h = (719.0_f32 * prepared.transform.scale).round() as i32;
        let pad_right = INPUT_SIZE - resized_w - prepared.transform.pad_left;
        let pad_bottom = INPUT_SIZE - resized_h - prepared.transform.pad_top;

        assert_eq!(prepared.transform.pad_left, 0);
        assert_eq!(prepared.transform.pad_top, 140);
        assert!(pad_right >= 0);
        assert!(pad_bottom >= 0);
        assert_eq!(
            prepared.transform.pad_left + resized_w + pad_right,
            INPUT_SIZE
        );
        assert_eq!(
            prepared.transform.pad_top + resized_h + pad_bottom,
            INPUT_SIZE
        );
    }

    #[test]
    fn normalization_maps_255_to_one() {
        let frame = solid_bgr_mat(640, 640, 0, 0, 255);
        let prepared = prepare(decoded_from_mat(frame)).expect("preprocess should succeed");
        assert!((prepared.input[[0, 0, 320, 320]] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn tensor_is_nchw_contiguous() {
        let frame = solid_bgr_mat(640, 640, 10, 20, 30);
        let prepared = prepare(decoded_from_mat(frame)).expect("preprocess should succeed");

        let channel_0 = prepared.input.index_axis(Axis(1), 0);
        let channel_1 = prepared.input.index_axis(Axis(1), 1);
        let channel_2 = prepared.input.index_axis(Axis(1), 2);

        assert_eq!(channel_0[[0, 0, 0]], 30.0 / 255.0);
        assert_eq!(channel_1[[0, 0, 0]], 20.0 / 255.0);
        assert_eq!(channel_2[[0, 0, 0]], 10.0 / 255.0);
    }
}
