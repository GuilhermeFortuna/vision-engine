use anyhow::Result;
use opencv::{
    core::{Point, Rect, Scalar},
    imgproc,
    prelude::*,
};

use vision_engine::detector::coco_class_name;
use vision_engine::tracking::{Track, TrackId, TrackState};

const LABEL_FONT_SCALE: f64 = 0.6;
const LABEL_THICKNESS: i32 = 1;
const LABEL_PADDING: i32 = 4;
const METRICS_AREA_RIGHT: i32 = 250;
const METRICS_AREA_BOTTOM: i32 = 160;

const CONFIRMED_BOX_THICKNESS: i32 = 2;
const TENTATIVE_BOX_THICKNESS: i32 = 1;

const COLOR_SATURATION: f64 = 0.8;
const COLOR_VALUE: f64 = 0.9;
const GOLDEN_RATIO: f64 = 0.618_033_988_75;

#[derive(Debug, Clone, Copy)]
pub struct FrameMetrics {
    pub decode_ms: f64,
    pub inference_ms: f64,
    pub tracking_ms: f64,
    pub fps: Option<f64>,
    pub confirmed_tracks: usize,
}

pub fn draw_tracks(frame: &mut Mat, tracks: &[Track]) -> Result<()> {
    let frame_w = frame.cols();
    let frame_h = frame.rows();
    let label_text_color = Scalar::new(0.0, 0.0, 0.0, 0.0);

    for track in tracks {
        if track.state == TrackState::Lost {
            continue;
        }

        let box_color = track_color(track.id);
        let label = label_text(track);
        let is_confirmed = track.state == TrackState::Confirmed;
        let box_thickness = if is_confirmed {
            CONFIRMED_BOX_THICKNESS
        } else {
            TENTATIVE_BOX_THICKNESS
        };

        let x_min = track.bbox.x_min.round() as i32;
        let y_min = track.bbox.y_min.round() as i32;
        let x_max = track.bbox.x_max.round() as i32;
        let y_max = track.bbox.y_max.round() as i32;

        let box_left = x_min.clamp(0, frame_w);
        let box_top = y_min.clamp(0, frame_h);
        let box_right = x_max.clamp(0, frame_w);
        let box_bottom = y_max.clamp(0, frame_h);

        if box_right <= box_left || box_bottom <= box_top {
            continue;
        }

        imgproc::rectangle(
            frame,
            Rect::new(
                box_left,
                box_top,
                box_right - box_left,
                box_bottom - box_top,
            ),
            box_color,
            box_thickness,
            imgproc::LINE_8,
            0,
        )?;

        let mut baseline = 0;
        let text_size = imgproc::get_text_size(
            &label,
            imgproc::FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            LABEL_THICKNESS,
            &mut baseline,
        )?;
        let text_w = text_size.width;
        let text_h = text_size.height;

        let bg_w = text_w + LABEL_PADDING * 2;
        let bg_h = text_h + LABEL_PADDING * 2;
        let (label_left, label_top) =
            label_origin(box_left, box_top, box_bottom, bg_w, bg_h, frame_w, frame_h);

        if is_confirmed {
            imgproc::rectangle(
                frame,
                Rect::new(label_left, label_top, bg_w, bg_h),
                box_color,
                imgproc::FILLED,
                imgproc::LINE_8,
                0,
            )?;
        }

        let text_origin = Point::new(
            label_left + LABEL_PADDING,
            label_top + LABEL_PADDING + text_h,
        );
        imgproc::put_text(
            frame,
            &label,
            text_origin,
            imgproc::FONT_HERSHEY_SIMPLEX,
            LABEL_FONT_SCALE,
            label_text_color,
            LABEL_THICKNESS,
            imgproc::LINE_8,
            false,
        )?;
    }

    Ok(())
}

pub fn draw_metrics_overlay(frame: &mut Mat, metrics: &FrameMetrics) -> Result<()> {
    let color = Scalar::new(0.0, 255.0, 0.0, 0.0);
    let font = imgproc::FONT_HERSHEY_SIMPLEX;
    let scale = 0.8;
    let thickness = 2;

    let decode_text = format!("Decode: {:.1} ms", metrics.decode_ms);
    imgproc::put_text(
        frame,
        &decode_text,
        Point::new(10, 30),
        font,
        scale,
        color,
        thickness,
        imgproc::LINE_8,
        false,
    )?;

    let inference_text = format!("Inference: {:.1} ms", metrics.inference_ms);
    imgproc::put_text(
        frame,
        &inference_text,
        Point::new(10, 60),
        font,
        scale,
        color,
        thickness,
        imgproc::LINE_8,
        false,
    )?;

    if let Some(fps) = metrics.fps {
        let fps_text = format!("FPS: {fps:.1}");
        imgproc::put_text(
            frame,
            &fps_text,
            Point::new(10, 90),
            font,
            scale,
            color,
            thickness,
            imgproc::LINE_8,
            false,
        )?;
    }

    let tracking_text = format!("Tracking: {:.1} ms", metrics.tracking_ms);
    imgproc::put_text(
        frame,
        &tracking_text,
        Point::new(10, 120),
        font,
        scale,
        color,
        thickness,
        imgproc::LINE_8,
        false,
    )?;

    let tracks_text = format!("Tracks: {}", metrics.confirmed_tracks);
    imgproc::put_text(
        frame,
        &tracks_text,
        Point::new(10, 150),
        font,
        scale,
        color,
        thickness,
        imgproc::LINE_8,
        false,
    )?;

    Ok(())
}

pub(crate) fn label_origin(
    box_left: i32,
    box_top: i32,
    box_bottom: i32,
    bg_w: i32,
    bg_h: i32,
    frame_w: i32,
    frame_h: i32,
) -> (i32, i32) {
    let mut label_left = box_left;
    let mut label_top = box_top - bg_h;

    if label_top < 0 {
        label_top = box_bottom;
    }

    if label_left + bg_w > frame_w {
        label_left = frame_w - bg_w;
    }
    if label_top + bg_h > frame_h {
        label_top = frame_h - bg_h;
    }
    if label_left < 0 {
        label_left = 0;
    }
    if label_top < 0 {
        label_top = 0;
    }

    let label_rect = Rect::new(label_left, label_top, bg_w, bg_h);
    if label_rect.x < METRICS_AREA_RIGHT && label_rect.y < METRICS_AREA_BOTTOM {
        label_top = box_bottom;
        if label_top + bg_h > frame_h {
            label_top = frame_h - bg_h;
        }
        if label_top < 0 {
            label_top = 0;
        }
    }

    (label_left, label_top)
}

pub(crate) fn track_color(id: TrackId) -> Scalar {
    let hue = (id.0 as f64 * GOLDEN_RATIO).fract() * 360.0;
    let (b, g, r) = hsv_to_bgr(hue, COLOR_SATURATION, COLOR_VALUE);
    Scalar::new(b, g, r, 0.0)
}

pub(crate) fn label_text(track: &Track) -> String {
    let class_name = coco_class_name(track.class_id).unwrap_or("unknown");
    match track.state {
        TrackState::Confirmed => format!("{class_name} {} {:.2}", track.id, track.confidence),
        TrackState::Tentative => format!("{class_name} ?"),
        TrackState::Lost => String::new(),
    }
}

fn hsv_to_bgr(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    let c = v * s;
    let h_prime = h / 60.0;
    let x = c * (1.0 - (h_prime % 2.0 - 1.0).abs());
    let m = v - c;
    let (r, g, b) = match h_prime as i32 {
        0 => (c, x, 0.0),
        1 => (x, c, 0.0),
        2 => (0.0, c, x),
        3 => (0.0, x, c),
        4 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };
    ((b + m) * 255.0, (g + m) * 255.0, (r + m) * 255.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vision_engine::tracking::clock::{FrameStamp, TimeSource};

    fn sample_stamp() -> FrameStamp {
        FrameStamp {
            index: 0,
            media_ms: 0.0,
            source: TimeSource::Reported,
            adjusted: false,
        }
    }

    fn sample_track(state: TrackState, id: u64, class_id: u32, confidence: f32) -> Track {
        let stamp = sample_stamp();
        Track {
            id: TrackId(id),
            class_id,
            state,
            bbox: vision_engine::tracking::BBox {
                x_min: 10.0,
                y_min: 10.0,
                x_max: 50.0,
                y_max: 50.0,
            },
            confidence,
            first_seen: stamp,
            last_seen: stamp,
            hits: 1,
            misses: 0,
        }
    }

    #[test]
    fn label_origin_top_edge_places_below_box() {
        let bg_w = 80;
        let bg_h = 20;
        let frame_w = 640;
        let frame_h = 480;
        let box_left = 100;
        let box_top = 0;
        let box_bottom = 40;

        let (left, top) = label_origin(box_left, box_top, box_bottom, bg_w, bg_h, frame_w, frame_h);

        assert_eq!(left, box_left);
        assert_eq!(top, box_bottom);
    }

    #[test]
    fn label_origin_bottom_edge_places_above_box() {
        let bg_w = 80;
        let bg_h = 20;
        let frame_w = 640;
        let frame_h = 480;
        let box_left = 100;
        let box_top = frame_h - 40;
        let box_bottom = frame_h;

        let (left, top) = label_origin(box_left, box_top, box_bottom, bg_w, bg_h, frame_w, frame_h);

        assert_eq!(left, box_left);
        assert_eq!(top, box_top - bg_h);
    }

    #[test]
    fn label_origin_right_edge_shifts_left() {
        let bg_w = 80;
        let bg_h = 20;
        let frame_w = 640;
        let frame_h = 480;
        let box_left = frame_w - 10;
        let box_top = 100;
        let box_bottom = 140;

        let (left, _) = label_origin(box_left, box_top, box_bottom, bg_w, bg_h, frame_w, frame_h);

        assert_eq!(left, frame_w - bg_w);
    }

    #[test]
    fn label_origin_metrics_area_pushes_below_box() {
        let bg_w = 80;
        let bg_h = 20;
        let frame_w = 640;
        let frame_h = 480;
        let box_left = 10;
        let box_top = 10;
        let box_bottom = 50;

        let (_, top) = label_origin(box_left, box_top, box_bottom, bg_w, bg_h, frame_w, frame_h);

        assert_eq!(top, box_bottom);
    }

    #[test]
    fn label_origin_oversized_label_yields_non_negative_origin() {
        let bg_w = 800;
        let bg_h = 20;
        let frame_w = 640;
        let frame_h = 480;
        let box_left = 100;
        let box_top = 100;
        let box_bottom = 140;

        let (left, top) = label_origin(box_left, box_top, box_bottom, bg_w, bg_h, frame_w, frame_h);

        assert!(left >= 0);
        assert!(top >= 0);
        assert_eq!(left, 0);
    }

    #[test]
    fn track_color_same_id_is_stable() {
        let id = TrackId(42);
        let first = track_color(id);
        let second = track_color(id);
        assert_eq!(first, second);
    }

    #[test]
    fn track_color_consecutive_ids_differ() {
        let c1 = track_color(TrackId(1));
        let c2 = track_color(TrackId(2));
        let diff_b = (c1[0] - c2[0]).abs();
        let diff_g = (c1[1] - c2[1]).abs();
        let diff_r = (c1[2] - c2[2]).abs();
        assert!(diff_b + diff_g + diff_r > 10.0);
    }

    #[test]
    fn track_color_channels_in_range() {
        for id in [0, 1, 42, 100, 999] {
            let color = track_color(TrackId(id));
            for channel in 0..3 {
                assert!(
                    (0.0..=255.0).contains(&color[channel]),
                    "channel {channel} for id {id}: {}",
                    color[channel]
                );
            }
        }
    }

    #[test]
    fn label_text_confirmed_includes_class_id_and_confidence() {
        let track = sample_track(TrackState::Confirmed, 42, 0, 0.91);
        assert_eq!(label_text(&track), "person #42 0.91");
    }

    #[test]
    fn label_text_tentative_omits_identity() {
        let track = sample_track(TrackState::Tentative, 42, 0, 0.91);
        assert_eq!(label_text(&track), "person ?");
    }
}
