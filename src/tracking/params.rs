/// A detection must survive three consecutive frames before its identity is
/// presented as stable.
pub const TRACK_PROMOTION_HITS: u32 = 3;

/// Roughly one second of occlusion tolerance, expressed in media time so
/// behavior does not change with source frame rate or with the unpaced loop.
pub const TRACK_RETENTION_MS: f64 = 1000.0;

/// Conventional SORT gate; retained pending VE-011 evidence.
pub const ASSOCIATION_IOU_GATE: f32 = 0.30;
