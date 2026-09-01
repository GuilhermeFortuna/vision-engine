# VE-010: Tracking visualization and metrics

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-009  
**Implementation plan:** [`../plans/VE-010-tracking-visualization-and-metrics-plan.md`](../plans/VE-010-tracking-visualization-and-metrics-plan.md)

## Purpose

Make tracking visible. Identities that exist only in memory cannot be reviewed or
accepted, so this pair renders tracks instead of raw detections and reports the
tracking-specific measurements Milestone 2 needs. It also moves rendering out of the
command-line entry point, which has accumulated three unrelated responsibilities.

## Requirements

### Rendering extraction

- Move the existing frame-drawing code out of the executable entry point into its
  own rendering module, preserving current behavior exactly.
- The entry point retains argument handling, startup validation, and the frame loop.
  It no longer owns drawing.
- This is a targeted extraction in service of this pair. Do not restructure the
  video, detection, or tracking modules, and do not convert the project to a
  workspace.

### Track rendering

- Render tracks rather than raw detections as the primary output.
- A confirmed track draws its box, its class name, its identity, and its confidence
  to two decimal places.
- A tentative track is drawn in a visually distinct, de-emphasized style and does not
  present an identity, because its identity is not yet stable.
- A lost track is not drawn.
- Each identity has a stable colour for as long as it lives, derived from the
  identity itself so that the same track keeps its colour across frames without
  storing a palette.
- Labels remain fully within the frame near every edge, and detection labels must not
  obscure the metrics overlay. VE-004's placement behavior is preserved.

### Metrics

- The overlay retains decode latency, inference latency, and processing frames per
  second from earlier pairs.
- The overlay adds tracking latency in milliseconds and the current confirmed track
  count.
- Tracking latency reports the measurement taken in VE-009 and does not re-time or
  re-derive it.
- Every displayed value describes measured work. No value is estimated, smoothed, or
  carried over from a previous frame without being marked as unavailable.

## Constraints and non-goals

- No trajectory trails, motion arrows, heat maps, zone overlays, or minimap.
- No interactive selection, track inspection panel, pause, or step controls.
- No output video encoding, screenshot capture, or clip extraction.
- No configurable colour scheme, theme, or command-line rendering options.
- No change to tracker behavior, thresholds, or lifecycle rules.

## Acceptance criteria

1. Rendering lives in its own module and the entry point no longer contains drawing
   code, with existing rendering behavior unchanged.
2. Confirmed tracks render with class name, identity, and two-decimal confidence.
3. Tentative tracks render de-emphasized without an identity, and lost tracks do not
   render.
4. A given identity keeps one colour for its lifetime, and the colour is derived from
   the identity rather than from allocation order.
5. Labels stay within frame bounds for tracks at every edge and corner, and the
   metrics overlay remains legible with many tracks on screen.
6. The overlay shows decode latency, inference latency, processing frames per second,
   tracking latency, and confirmed track count simultaneously.
7. Pure logic, including colour derivation and label placement, is unit tested
   without requiring a display.
8. Formatting, linting, tests, and the release build pass.
