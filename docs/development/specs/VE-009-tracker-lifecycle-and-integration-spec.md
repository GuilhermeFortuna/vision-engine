# VE-009: Tracker lifecycle and pipeline integration

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-008  
**Implementation plan:** [`../plans/VE-009-tracker-lifecycle-and-integration-plan.md`](../plans/VE-009-tracker-lifecycle-and-integration-plan.md)

## Purpose

Combine the motion model and the association step into a tracker that owns identity
over time, and place it in the frame loop. Completing this pair is what makes
Milestone 2 real: the same object keeps the same identity across frames, and
identities are born and retired on defined rules.

## Requirements

### Per-frame sequence

- The tracker exposes one entry point that accepts the current frame's detections
  and its frame stamp, and returns the tracks for that frame.
- Each call performs, in order: predict every live track, associate predictions with
  detections, apply matched updates, age unmatched tracks, spawn tracks from
  unmatched detections, and retire tracks that exceed the retention limit.
- Prediction happens exactly once per track per frame, and association consumes
  predicted boxes rather than last-matched boxes.

### Identity and state transitions

- An unmatched detection spawns a tentative track with a fresh identity, its
  first-seen and last-seen stamps set to the current frame.
- A tentative track that accumulates the promotion threshold in hits becomes
  confirmed. A tentative track that misses a frame before reaching that threshold is
  retired immediately, so spurious detections do not accumulate identities.
- A matched track updates its filter, its box, its class-consistent confidence from
  the matched detection, its last-seen stamp, and its hit count, and resets its
  consecutive-miss count.
- An unmatched confirmed track keeps its predicted box, increments its miss count,
  and remains eligible for matching until its retention limit elapses.
- A track whose time since last match exceeds the media-time retention limit becomes
  lost and is removed from the live set. A retired identity is never reissued.
- A rejected filter update from VE-007 does not remove the track and does not fail
  the frame. The track keeps its prediction, ages normally, and the occurrence is
  counted.

### Bounded state

- The live track collection is pruned every frame. Its size must be a function of
  current scene content, not of how long the process has been running.
- No per-frame history, trajectory buffer, detection archive, or event log is
  retained. Tracks carry first-seen and last-seen stamps only.
- A run over a long video must not grow the live track collection without bound.
  This is the pair's primary memory risk and is verified rather than assumed.

### Pipeline integration

- The tracker is constructed once per process run, before the frame loop, and reused
  for every frame.
- Tracking runs after post-processing and before rendering, on the same thread, in
  the existing synchronous loop.
- Tracking duration is measured per frame around the tracker call only, separate
  from decode and inference timing.
- A tracker failure carries context identifying the tracking stage and the frame
  index, consistent with the repository's error handling.

## Constraints and non-goals

- No rendering change, overlay change, or label change. VE-010 owns visualization.
- No events, zones, lines, dwell time, counting, persistence, or search.
- No cross-camera identity, appearance embeddings, or re-identification.
- No threading, queues, or pipeline restructuring; Milestone 3 owns parallelism.
- Do not tune detection thresholds or the model contract to improve tracking.

## Acceptance criteria

1. A deterministic synthetic sequence of detections produces exactly the expected
   identities, asserted by value across frames, including through a gap shorter than
   the retention limit.
2. An object absent longer than the retention limit is retired, and its later
   reappearance receives a new identity rather than the retired one.
3. A tentative track that misses before promotion is discarded and never becomes
   confirmed; a track reaching the promotion threshold becomes confirmed on the
   expected frame.
4. First-seen and last-seen stamps match the frames on which the track was actually
   first and most recently matched.
5. Prediction is invoked once per track per frame and association operates on
   predicted boxes, demonstrated by a test where the two differ.
6. A rejected filter update leaves the track present and the frame successful, and
   increments the recorded count.
7. Over a long synthetic run with objects entering and leaving, the live track
   collection returns to a bounded size and does not grow with frame count.
8. The tracker is constructed once per run and tracking latency is measured
   separately from decode and inference.
9. Tests are pure and headless, requiring no video, model, or display.
10. Formatting, linting, tests, and the release build pass.
