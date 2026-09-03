use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::tracking::clock::FrameStamp;
use crate::tracking::{Track, TrackState};
use anyhow::{Context, Result};

const HEADER: &str =
    "frame_index,media_ms,track_id,class_id,state,x_min,y_min,x_max,y_max,confidence";

pub struct TrackDump {
    writer: BufWriter<File>,
}

fn state_name(state: TrackState) -> &'static str {
    match state {
        TrackState::Confirmed => "confirmed",
        TrackState::Tentative => "tentative",
        TrackState::Lost => "lost",
    }
}

fn format_track_line(stamp: FrameStamp, track: &Track) -> String {
    format!(
        "{},{:.3},{},{},{},{:.3},{:.3},{:.3},{:.3},{:.4}",
        stamp.index,
        stamp.media_ms,
        track.id.0,
        track.class_id,
        state_name(track.state),
        track.bbox.x_min,
        track.bbox.y_min,
        track.bbox.x_max,
        track.bbox.y_max,
        track.confidence,
    )
}

impl TrackDump {
    pub fn create(path: &Path) -> Result<Self> {
        let file = File::create(path)
            .with_context(|| format!("failed to create track dump at {}", path.display()))?;
        let mut writer = BufWriter::new(file);
        writeln!(writer, "{HEADER}")?;
        Ok(Self { writer })
    }

    pub fn write_frame(&mut self, stamp: FrameStamp, tracks: &[Track]) -> Result<()> {
        let mut sorted: Vec<&Track> = tracks.iter().collect();
        sorted.sort_by_key(|track| track.id);
        for track in sorted {
            writeln!(self.writer, "{}", format_track_line(stamp, track))?;
        }
        Ok(())
    }

    pub fn finish(mut self) -> Result<()> {
        self.writer.flush().context("failed to flush track dump")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tracking::clock::TimeSource;
    use crate::tracking::{BBox, TrackId};
    use std::io::Read;

    fn sample_stamp(index: u64, media_ms: f64) -> FrameStamp {
        FrameStamp {
            index,
            media_ms,
            source: TimeSource::Reported,
            adjusted: false,
        }
    }

    fn sample_track(
        id: u64,
        state: TrackState,
        class_id: u32,
        confidence: f32,
        bbox: BBox,
    ) -> Track {
        let stamp = sample_stamp(0, 0.0);
        Track {
            id: TrackId(id),
            class_id,
            state,
            bbox,
            confidence,
            first_seen: stamp,
            last_seen: stamp,
            hits: 1,
            misses: 0,
        }
    }

    fn dump_lines(path: &Path) -> Vec<String> {
        let mut file = File::open(path).expect("failed to open dump file");
        let mut contents = String::new();
        file.read_to_string(&mut contents)
            .expect("failed to read dump file");
        contents.lines().map(str::to_string).collect()
    }

    #[test]
    fn confirmed_track_line_format_is_exact() {
        let dir = std::env::temp_dir().join(format!(
            "vision-engine-track-dump-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("dump.csv");

        let stamp = sample_stamp(7, 233.333);
        let track = sample_track(
            42,
            TrackState::Confirmed,
            0,
            0.91,
            BBox {
                x_min: 10.0,
                y_min: 20.0,
                x_max: 110.0,
                y_max: 220.0,
            },
        );

        let mut dump = TrackDump::create(&path).expect("failed to create dump");
        dump.write_frame(stamp, &[track])
            .expect("failed to write frame");
        dump.finish().expect("failed to finish dump");

        let lines = dump_lines(&path);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], HEADER);
        assert_eq!(
            lines[1],
            "7,233.333,42,0,confirmed,10.000,20.000,110.000,220.000,0.9100"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn tracks_are_emitted_in_ascending_id_order() {
        let dir = std::env::temp_dir().join(format!(
            "vision-engine-track-dump-order-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|duration| duration.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).expect("failed to create temp dir");
        let path = dir.join("dump.csv");

        let stamp = sample_stamp(1, 33.333);
        let bbox = BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 10.0,
            y_max: 10.0,
        };
        let track_high = sample_track(99, TrackState::Tentative, 1, 0.5, bbox);
        let track_low = sample_track(7, TrackState::Confirmed, 0, 0.8, bbox);

        let mut dump = TrackDump::create(&path).expect("failed to create dump");
        dump.write_frame(stamp, &[track_high, track_low])
            .expect("failed to write frame");
        dump.finish().expect("failed to finish dump");

        let lines = dump_lines(&path);
        assert_eq!(lines.len(), 3);
        assert!(lines[1].starts_with("1,33.333,7,"));
        assert!(lines[2].starts_with("1,33.333,99,"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
