# VE-008: Detection to track association

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-007  
**Implementation plan:** [`../plans/VE-008-detection-track-association-plan.md`](../plans/VE-008-detection-track-association-plan.md)

## Purpose

Decide which detection belongs to which existing track on a given frame. This pair
delivers the assignment step as a pure, deterministic function over predicted track
boxes and current detections, so the lifecycle pair that consumes it can be written
without also solving the matching problem.

## Requirements

### Class partitioning

- Association is performed independently within each class. Person tracks are
  matched only against person detections, car tracks only against car detections,
  and so on.
- Partitioning is done by grouping before the solver runs. Cross-class pairs are
  never presented to the solver, and artificial or infinite costs are never used to
  suppress them. Feeding unreachable sentinel costs into an assignment solver is
  explicitly rejected as an implementation strategy.
- Classes present only among tracks, or only among detections, still produce correct
  unmatched output for their side.

### Cost and assignment

- The association cost between a predicted track box and a detection box is derived
  from their intersection over union, using the shared implementation from VE-006.
- Within each class, compute a minimum-cost one-to-one assignment between predicted
  track boxes and detection boxes.
- Rectangular problems are supported in both directions: more tracks than
  detections, and more detections than tracks. Empty input on either side is valid
  and yields the appropriate unmatched output.
- After solving, a candidate pair whose overlap falls below the association gate is
  rejected and both of its members are returned as unmatched. Gating is applied
  after assignment, not by excluding pairs beforehand.

### Output contract

- The result reports three disjoint groups: matched track-and-detection pairs, track
  indices with no match, and detection indices with no match.
- Every input track and every input detection appears exactly once across the three
  groups. This total-coverage property is a tested invariant.
- Results are deterministic. Equal-cost alternatives resolve by lowest index so that
  repeated runs over identical input produce identical assignments.
- Indices in the result refer to positions in the caller's original, unpartitioned
  input slices, so the caller does not have to reverse the grouping.

### Dependency posture

- Use a small, focused linear-assignment implementation. Do not adopt a broad graph
  or pathfinding library for the sake of one assignment routine.
- Whether to take a dependency at all or implement the solver directly is decided in
  the plan against the repository's dependency policy, after checking maintenance,
  API, and license. Expected problem sizes are small, so a self-contained
  implementation is a legitimate outcome. This specification does not name a crate.

## Constraints and non-goals

- No track state changes, identity allocation, promotion, retirement, or filter
  updates. This pair only reports who matches whom.
- No second-stage or low-confidence recovery association, no appearance features, no
  re-identification, no motion gating beyond the overlap gate.
- No parallelism. No unsafe code.

## Acceptance criteria

1. A clean one-to-one scenario matches every track to its correct detection.
2. Surplus detections and surplus tracks are reported as unmatched, in both
   directions, including when one side is empty.
3. Tracks and detections of different classes are never matched, and each class is
   solved independently.
4. A pair whose overlap is below the gate is rejected after assignment and both
   members are returned as unmatched.
5. Matched, unmatched-track, and unmatched-detection groups always account for every
   input exactly once.
6. Identical input produces identical output across repeated runs, including when
   costs are tied.
7. Returned indices address the caller's original input order.
8. Tests are pure and headless, and cover the assignment solver's own correctness on
   cases where a greedy match would give the wrong answer.
9. Formatting, linting, tests, and the release build pass.
