use anyhow::{Context, Result};
use opencv::{
    core::{Point, Rect, Scalar},
    highgui, imgproc,
    prelude::*,
};

use super::message::TrackedFrame;
use super::metrics::{FrameMetrics, format_depth, format_ms};
use crate::detector::coco_class_name;
use crate::tracking::{Track, TrackId, TrackState};

const WINDOW_NAME: &str = "vision-engine";
const LABEL_FONT_SCALE: f64 = 0.6;
const LABEL_THICKNESS: i32 = 1;
const LABEL_PADDING: i32 = 4;
const METRICS_AREA_RIGHT: i32 = 250;
const METRICS_AREA_BOTTOM: i32 = 280;

const CONFIRMED_BOX_THICKNESS: i32 = 2;
const TENTATIVE_BOX_THICKNESS: i32 = 1;

const COLOR_SATURATION: f64 = 0.8;
const COLOR_VALUE: f64 = 0.9;
const GOLDEN_RATIO: f64 = 0.618_033_988_75;

pub struct RenderStage {}

pub enum Presentation {
    Continue,
    QuitRequested,
}

impl RenderStage {
    pub fn open() -> Result<Self> {
        highgui::named_window(WINDOW_NAME, highgui::WINDOW_AUTOSIZE)
            .context("failed to create display window")?;
        Ok(Self {})
    }

    pub fn present(
        &mut self,
        mut tracked: TrackedFrame,
        metrics: &FrameMetrics,
    ) -> Result<Presentation> {
        draw_tracks(&mut tracked.frame, &tracked.tracks)
            .context("failed to draw track overlays")?;
        draw_metrics_overlay(&mut tracked.frame, metrics)
            .context("failed to draw performance metrics overlay")?;
        highgui::imshow(WINDOW_NAME, &tracked.frame).context("failed to display video frame")?;

        let key = highgui::wait_key(1).context("failed to poll keyboard events")?;
        if should_exit(key) {
            Ok(Presentation::QuitRequested)
        } else {
            Ok(Presentation::Continue)
        }
    }

    pub fn close(self) -> Result<()> {
        highgui::destroy_window(WINDOW_NAME).context("failed to destroy display window")
    }
}

fn should_exit(key: i32) -> bool {
    if key == -1 {
        return false;
    }

    let key = key & 0xFF;
    key == 27 || key == i32::from(b'q') || key == i32::from(b'Q')
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
    let line_height = 30;
    let mut y = 30;

    let lines = [
        format!("Decode: {} ms", format_ms(Some(metrics.timings.decode_ms))),
        format!(
            "Preprocess: {} ms",
            format_ms(Some(metrics.timings.preprocess_ms))
        ),
        format!(
            "Inference: {} ms",
            format_ms(Some(metrics.timings.inference_ms))
        ),
        format!(
            "Tracking: {} ms",
            format_ms(Some(metrics.timings.tracking_ms))
        ),
        format!(
            "Render (prev): {} ms",
            format_ms(metrics.render_ms.is_finite().then_some(metrics.render_ms))
        ),
        format!(
            "Decoded: {}",
            format_depth(
                metrics.queue_depths.decoded.0,
                metrics.queue_depths.decoded.1
            )
        ),
        format!(
            "Prepared: {}",
            format_depth(
                metrics.queue_depths.prepared.0,
                metrics.queue_depths.prepared.1
            )
        ),
        format!(
            "Detected: {}",
            format_depth(
                metrics.queue_depths.detected.0,
                metrics.queue_depths.detected.1
            )
        ),
        format!(
            "Tracked: {}",
            format_depth(
                metrics.queue_depths.tracked.0,
                metrics.queue_depths.tracked.1
            )
        ),
        format!(
            "Renderer throughput: {} fps",
            match metrics.fps {
                Some(fps) if fps.is_finite() => format!("{fps:.1}"),
                _ => "--".to_string(),
            }
        ),
        format!("Tracks: {}", metrics.confirmed_tracks),
    ];

    for line in lines {
        imgproc::put_text(
            frame,
            &line,
            Point::new(10, y),
            font,
            scale,
            color,
            thickness,
            imgproc::LINE_8,
            false,
        )?;
        y += line_height;
    }

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
    use crate::tracking::clock::{FrameStamp, TimeSource};

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
            bbox: crate::tracking::BBox {
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
    fn escape_exits() {
        assert!(should_exit(27));
    }

    #[test]
    fn lowercase_q_exits() {
        assert!(should_exit(113));
    }

    #[test]
    fn uppercase_q_exits() {
        assert!(should_exit(81));
    }

    #[test]
    fn no_key_continues() {
        assert!(!should_exit(-1));
    }

    #[test]
    fn other_key_continues() {
        assert!(!should_exit(65));
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
