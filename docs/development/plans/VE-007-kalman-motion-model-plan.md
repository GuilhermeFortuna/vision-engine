# VE-007 implementation plan: Kalman motion model

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-007-kalman-motion-model-spec.md`](../specs/VE-007-kalman-motion-model-spec.md)  
**Depends on:** VE-006

## Current-system context

VE-006 provides `BBox` and the `src/tracking/` module. Nothing in the repository
performs prediction today. This pair adds one self-contained filter file and one
dependency, and touches no other module.

## Interfaces produced

```rust
// src/tracking/kalman.rs
pub enum RejectReason { SingularCovariance, NonFiniteState }
pub enum UpdateOutcome { Applied, Rejected(RejectReason) }

pub struct KalmanBoxTracker { /* private: x: SVector<f32,7>, p: SMatrix<f32,7,7> */ }

impl KalmanBoxTracker {
    pub fn new(bbox: &BBox) -> Self;
    pub fn predict(&mut self) -> BBox;          // advances state, returns predicted box
    pub fn update(&mut self, bbox: &BBox) -> UpdateOutcome;
    pub fn bbox(&self) -> BBox;                 // current state as a box
    pub fn rejected_updates(&self) -> u64;
}
```

## Implementation decisions

- Add `nalgebra` with default features. Justification for the task summary: the
  update step needs a numerically sound solve of a 4x4 system, and hand-rolling that
  is exactly the kind of error-prone numerical code the dependency policy exists to
  avoid. Fixed-size `SMatrix`/`SVector` types keep the filter allocation-free.
- State is `[cx, cy, s, r, vcx, vcy, vs]` where `s` is box area and `r` is aspect
  ratio; measurement is `[cx, cy, s, r]`. Aspect ratio is modeled as constant, which
  is why it has no velocity term.
- Matrices, built once as constants or in `new`: `F` is the 7x7 identity with
  `F[(0,4)] = F[(1,5)] = F[(2,6)] = 1.0`. `H` is 4x7 selecting the first four state
  elements. Initial `P` is diagonal, scaled by 10, with the three velocity entries
  scaled by a further 1000 to express that velocity is unobserved at birth. `Q` is
  the identity with the velocity block scaled by 0.01 and the area-velocity entry by
  a further 0.01. `R` is the identity with the area and aspect entries scaled by 10,
  since those measurements are noisier than centre position.
- **The update solves rather than inverts.** With `M = P * Hᵀ` (7x4) and
  `S = H * P * Hᵀ + R` (4x4), the gain satisfies `K * S = M`. Because `S` is
  symmetric, solve `S * Xᵀ = Mᵀ` with `S.lu().solve(&m.transpose())` and take
  `K = X.transpose()`. Do not call `try_inverse`. A `None` from `solve` is
  `RejectReason::SingularCovariance`.
- On a rejected update the filter mutates nothing: the state stays exactly as
  `predict` left it, `rejected_updates` increments, and the caller decides what to
  do. This is what keeps one bad track from failing a frame.
- Guard non-finite values on both sides. `update` checks the incoming box with
  `BBox::is_valid` and checks the post-update state and covariance for finiteness;
  if the post-update values are non-finite, roll back to the pre-update state and
  return `Rejected(NonFiniteState)`. Take a copy of `x` and `P` before applying the
  gain so the rollback is exact.
- `predict` clamps area to `MIN_AREA = 1.0`. When the clamp fires, also zero the
  area velocity so the filter does not keep driving the box toward collapse on
  subsequent predictions.
- `bbox()` and the state conversion derive width as `(s * r).sqrt()` and height as
  `s / width`, guarding a non-positive or non-finite width by falling back to
  `MIN_AREA` dimensions.

## Ordered implementation

1. Add `nalgebra` to `Cargo.toml` and create `src/tracking/kalman.rs`.
2. Write failing tests for the box-to-state-to-box round trip across landscape,
   portrait, and square boxes, asserting agreement within `1e-3` relative.
3. Implement `new`, the state conversions, and `bbox` until those pass.
4. Write a failing test that constructs a filter from a box, then alternates
   `predict` and `update` along a straight constant-velocity path for eight frames,
   and asserts the ninth `predict` lands within a stated pixel tolerance of the true
   position with no update supplied.
5. Write a failing test that a stationary box stays within tolerance of its origin
   over twenty predict-update cycles.
6. Implement `predict` and `update`, using the solve described above, until both
   pass.
7. Write a failing test asserting the covariance trace increases across three
   consecutive `predict` calls with no update, and decreases after an `update`.
8. Write failing tests for the rejection paths: a degenerate measurement that yields
   a singular system returns `Rejected(SingularCovariance)`, a non-finite input box
   returns `Rejected(NonFiniteState)`, both leave `bbox()` equal to the predicted box
   and both increment `rejected_updates`.
9. Write a failing test that fifty consecutive `predict` calls with no update never
   produce a non-positive area, a non-finite coordinate, or an invalid box.
10. Implement the guards until every rejection and stability test passes.
11. Grep the finished module to confirm no `try_inverse`, no `unwrap`, no `expect`,
    and no OpenCV or `ort` reference appears in it.
12. Run the full validation suite.

## Validation

- Unit: round-trip conversion, constant-velocity prediction accuracy, stationary
  stability, covariance growth and shrink, both rejection reasons, rollback
  exactness, long-run area and finiteness stability.
- Static: the module compiles with no reference to OpenCV, ONNX Runtime, or the
  frame loop, and contains no explicit inverse.
- All tests are pure and require no video, model, or display.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```

## Handoff

Report the tolerances chosen for the prediction and round-trip tests and why, the
observed covariance behavior, how the singular case was constructed, and confirmation
that the update path solves rather than inverts. Note the `nalgebra` version and the
justification recorded for it.
