# VE-008 implementation plan: Detection to track association

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-008-detection-track-association-spec.md`](../specs/VE-008-detection-track-association-spec.md)  
**Depends on:** VE-007

## Current-system context

VE-006 supplies `BBox::iou` and `ASSOCIATION_IOU_GATE`; VE-007 supplies predicted
boxes. Nothing matches detections to tracks yet. This pair is a pure function over
two slices and introduces at most one dependency.

## Interfaces produced

```rust
// src/tracking/assignment.rs
pub struct Association {
    pub matches: Vec<(usize, usize)>,        // (track index, detection index)
    pub unmatched_tracks: Vec<usize>,
    pub unmatched_detections: Vec<usize>,
}

/// `tracks` is (class_id, predicted box) in the caller's order.
/// Returned indices address `tracks` and `detections` as given.
pub fn associate(
    tracks: &[(u32, BBox)],
    detections: &[Detection],
    iou_gate: f32,
) -> Association;
```

## Implementation decisions

- **Solver sourcing is decided in step 1, not assumed here.** Evaluate candidate
  linear-assignment crates against the dependency policy: last release within
  eighteen months, an API that accepts a rectangular cost matrix or is trivially
  padded, a permissive licence compatible with the project, and no transitive graph
  or pathfinding framework. Record the crates examined and the verdict in the task
  summary. If none clears the bar, implement Kuhn-Munkres directly in this module.
  Either outcome satisfies the spec; a broad graph library pulled in for one routine
  does not.
- If implementing directly: the square-padded Kuhn-Munkres form is sufficient.
  Problem sizes here are bounded by the detections surviving VE-004's threshold on a
  single frame, which is tens of boxes, so an O(n^3) implementation on a padded
  matrix is comfortably fast and far easier to test than an optimised sparse variant.
  Pad the shorter dimension with a cost strictly greater than any real cost, and
  discard pad pairs when reading the result.
- **Partition before solving.** Group track indices and detection indices by class
  into per-class index lists, then run the solver once per class on a matrix built
  only from that class's members. Cross-class pairs never enter a cost matrix, so no
  sentinel or infinite cost is ever constructed. A class present on only one side
  skips the solver entirely and contributes all of its indices to the corresponding
  unmatched list.
- Cost is `1.0 - a.iou(b)`, which lies in `[0, 1]` for valid boxes. Any pair whose
  IoU is not finite is treated as cost `1.0`, so malformed geometry can never win an
  assignment.
- **Gate after solving.** Run the solver on the ungated matrix, then walk the
  resulting pairs and move any pair whose IoU is at or below `iou_gate` into the two
  unmatched lists. Gating beforehand by removing columns would change which
  assignment is optimal, which is the bug this ordering avoids.
- Determinism: iterate classes in ascending class id, build each matrix in ascending
  original index order, and have the solver break equal-cost choices toward the
  lowest index. Sort `matches` by track index and both unmatched lists ascending
  before returning, so output is a pure function of input with no map-iteration
  order leaking in.
- Index translation happens once at the end: per-class local indices map back
  through the grouping lists to the caller's original positions.

## Ordered implementation

1. Evaluate assignment crates against the criteria above; record the decision and
   add the dependency, or create the in-repo solver module.
2. Write a failing test for the solver alone on a 3x3 cost matrix whose greedy
   nearest-choice answer is provably worse than the optimal assignment, asserting
   the optimal permutation. This is the test that proves a real solver is present
   rather than a greedy loop.
3. Write failing solver tests for rectangular inputs in both directions, and for a
   matrix with tied costs asserting the lowest-index resolution.
4. Implement or wire the solver until those pass.
5. Write a failing test for `associate` on one class, three tracks and three
   detections in clean one-to-one correspondence, asserting exact pairs.
6. Write failing tests for surplus detections, surplus tracks, an empty detection
   slice, and an empty track slice, each asserting that the three result groups
   together account for every input index exactly once.
7. Write a failing test with a person track overlapping a car detection at high IoU
   and asserting they are never matched, plus a case where two classes are each
   solved correctly in the same call.
8. Write a failing test where the optimal assignment pairs boxes below the gate,
   asserting both members appear as unmatched and the pair is absent.
9. Write a failing test asserting byte-identical results across ten repeated calls
   on the same input, including a tied-cost case.
10. Implement `associate` until all tests pass.
11. Add a total-coverage property assertion as a helper used by every `associate`
    test, so the invariant is checked everywhere rather than in one case.
12. Run the full validation suite.

## Validation

- Unit: solver optimality against a greedy-defeating matrix, rectangular and empty
  inputs, class partitioning, post-solve gating, deterministic tie-breaking, index
  translation to caller order, and total coverage on every case.
- Static: no sentinel or infinite cost value appears in the module.
- All tests are pure and require no video, model, or display.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```

## Handoff

Report the crates evaluated with the verdict for each and the sourcing decision
taken, the greedy-defeating matrix used to prove optimality, and confirmation that
gating runs after assignment and that no cross-class pair is ever constructed.
