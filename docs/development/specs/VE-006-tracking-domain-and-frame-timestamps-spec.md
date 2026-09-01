# VE-006: Tracking domain model and frame timestamps

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-005  
**Implementation plan:** [`../plans/VE-006-tracking-domain-and-frame-timestamps-plan.md`](../plans/VE-006-tracking-domain-and-frame-timestamps-plan.md)

## Purpose

Establish the shared temporal and spatial vocabulary Milestone 2 needs before any
tracking behavior exists. This pair anchors every decoded frame to a defined
timestamp and introduces the bounding-box, identity, and track types the later
pairs operate on, so association and lifecycle work does not also have to invent
its domain model.

## Requirements

### Frame timestamps

- Every successfully decoded frame produces one frame stamp carrying a zero-based
  frame index and a media-time offset in milliseconds measured from the start of
  the source.
- The frame index is authoritative for ordering, determinism, and test assertions.
  Media time is descriptive and may be approximate.
- Media time is defined semantically: it is the presentation time of the frame that
  was just decoded, expressed in the source's own timeline. The specific decoder
  property and the order in which it is read are implementation details chosen in
  the plan, not part of this contract. Backend timestamp semantics vary between
  OpenCV capture backends and must not be assumed.
- When the decoder reports no usable media time, derive it from the frame index and
  the source's reported frame rate. When the reported frame rate is also unusable,
  derive it from the frame index at a stated nominal rate.
- Each frame stamp records how its media time was obtained, distinguishing at
  minimum: reported by the decoder, derived from the source frame rate, and derived
  from the frame index alone. Downstream consumers and acceptance evidence must be
  able to tell measured time from synthesized time.
- Media time must be non-decreasing across a run. A regression is corrected, the
  corrected stamp is marked as adjusted, and the number of adjustments is counted
  and reported at least once per run. Regressions are never silently repaired.

### Shared spatial type

- Define one axis-aligned floating-point bounding-box type in source-frame
  coordinates, shared by detections and tracks.
- It provides the geometry later pairs require: corner access, center and size
  access, area, and intersection over union.
- `Detection` adopts this type so the repository does not carry two bounding-box
  representations or two intersection-over-union implementations. VE-004's existing
  numeric behavior and its unit tests are preserved unchanged.

### Track identity and state

- Track identity is a distinct, copyable, displayable value allocated in increasing
  order and never reused within a process run.
- Track state distinguishes tentative (seen, not yet stable), confirmed (stable
  identity suitable for display and later events), and lost (retired).
- A track carries: identity, class id, state, current bounding box, confidence,
  first-seen and last-seen frame stamps, cumulative hit count, and consecutive
  misses since its last match.
- Track confidence is the confidence of the most recently matched detection. This
  milestone introduces no smoothing, decay, or derived confidence model.

### Tracker parameters

- Promotion threshold, retention limit, and association gate are named constants
  declared in one place, each carrying a documented justification rather than an
  inherited default.
- Retention is expressed in media time rather than frame count so behavior does not
  change with source frame rate, and because playback is unpaced.
- No command-line or configuration-file surface is added for these values.

## Constraints and non-goals

- No motion model, prediction, association, lifecycle transition, tracker struct,
  rendering change, event emission, or persistence.
- No change to detection thresholds, model contract, or the existing playback loop
  beyond producing and carrying the frame stamp.
- No seeking, reverse playback, or variable-rate playback support.
- Do not restructure the repository beyond introducing the tracking module.

## Acceptance criteria

1. Every decoded frame yields a frame stamp whose index increases by exactly one
   per successfully decoded frame, starting at zero.
2. Media time is non-decreasing for a representative local video, and each stamp
   reports the provenance of its media time.
3. A source that reports no usable media time still produces non-decreasing,
   frame-rate-derived stamps, and the provenance reflects the fallback used.
4. Timestamp regressions are corrected, marked as adjusted, counted, and reported;
   no regression is discarded without a record.
5. Detections and tracks share one bounding-box type and one intersection-over-union
   implementation, and VE-004's detection tests still assert the same properties.
6. Track identities are unique and increasing within a run, and track state
   distinguishes tentative, confirmed, and lost.
7. Tracker parameters are named constants with recorded justification, and retention
   is expressed in media time.
8. Formatting, linting, tests, and the release build pass.
