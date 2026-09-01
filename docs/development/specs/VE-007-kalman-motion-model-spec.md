# VE-007: Kalman motion model

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-006  
**Implementation plan:** [`../plans/VE-007-kalman-motion-model-plan.md`](../plans/VE-007-kalman-motion-model-plan.md)

## Purpose

Give each track a motion model so its position can be predicted for the frame
currently being processed rather than compared against where it was last seen.
This is what allows identities to survive brief occlusion and fast movement. The
pair delivers the filter as a self-contained, numerically tested unit with no
decoding, inference, or association in it.

## Requirements

### Filter contract

- Implement a linear Kalman filter with a constant-velocity motion model over a
  seven-element state: box center x, center y, area, aspect ratio, and the
  velocities of center x, center y, and area. Aspect ratio is modeled as constant.
- The measurement is the four-element observable part of that state: center x,
  center y, area, and aspect ratio.
- Convert between the shared bounding-box type from VE-006 and both the state and
  measurement representations. Conversion round-trips within a documented tolerance.
- Initialize a filter from a single bounding box, with low uncertainty on observed
  quantities and high uncertainty on the unobserved velocities.

### Prediction and update

- Prediction advances the state by one time step and increases uncertainty. It is
  called once per track per frame before association.
- Area must remain strictly positive after prediction. A prediction that would
  drive area to zero or below is clamped to a positive floor so the resulting box
  stays well formed.
- Prediction exposes the predicted bounding box for association to consume.
- Update corrects the state from a matched measurement and reduces uncertainty.
- The update must solve the resulting linear system rather than forming an explicit
  inverse of the innovation covariance. The numerical method is delegated to the
  linear-algebra dependency rather than hand-implemented.

### Numerical failure behavior

- An update whose innovation covariance cannot be solved is reported to the caller
  as a recoverable, non-fatal outcome, not as a fatal error and not as a panic.
- A non-finite value appearing anywhere in the state or covariance is detected and
  reported through the same recoverable outcome rather than propagating silently
  into predicted boxes.
- The filter defines what happens to its own state on a rejected update: the state
  is left as predicted, so the track ages naturally instead of being corrupted.
- The module counts rejected updates so a persistently failing filter is
  distinguishable from an isolated numerical event. A count is required; a log line
  alone is not sufficient.

### Isolation

- The module depends only on the linear-algebra crate and VE-006's domain types.
  It must not reference OpenCV, ONNX Runtime, the video loop, or rendering.
- Matrix dimensions are fixed and known at compile time. No dynamic allocation is
  required per prediction or update.

## Constraints and non-goals

- No association, cost matrices, track lifecycle, identity allocation, or
  multi-track container. This pair models one track's motion and nothing else.
- No non-linear filter, no unscented or extended variant, no motion model beyond
  constant velocity, and no per-class tuning.
- No parallelism, SIMD, or unsafe code. No optimization without a measured baseline.
- Do not add a general-purpose matrix or statistics abstraction.

## Acceptance criteria

1. A box moving at constant velocity is predicted within a stated tolerance several
   frames ahead from measurements alone.
2. A stationary box stays stationary under repeated predict-and-update cycles.
3. Uncertainty grows monotonically across consecutive predictions without updates,
   and shrinks on update.
4. Bounding box to state to bounding box round-trips within the documented
   tolerance for landscape, portrait, and square boxes.
5. Prediction never produces a non-positive area or a malformed box, including
   after many predictions without an update.
6. An unsolvable innovation covariance and a non-finite state both yield the
   recoverable outcome, leave the predicted state intact, increment the rejection
   count, and never panic.
7. The update path contains no explicit inverse of the innovation covariance.
8. Tests are pure and headless, requiring no video, model, or display.
9. Formatting, linting, tests, and the release build pass.
