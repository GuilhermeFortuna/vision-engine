# VE-014: Threaded pipeline runtime

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-013  
**Implementation plan:** [`../plans/VE-014-threaded-pipeline-runtime-plan.md`](../plans/VE-014-threaded-pipeline-runtime-plan.md)

## Purpose

Run the pipeline concurrently. Decode, preprocess, inference, and tracking each get a
dedicated thread connected by the bounded queues from VE-013, and rendering stays on
the main thread because the display library requires it. This is where Milestone 3's
throughput comes from, and where its risk lives: the run must produce exactly the
identities the serial baseline produced, and it must always terminate.

## Requirements

### Threading and ownership

- Decode, preprocess, inference, and tracking each run on one dedicated thread.
  Rendering runs on the main thread.
- No stage runs more than one thread. No worker pools, no reorder buffers.
- Each stage exclusively owns its resources: the decoder owns the video capture, the
  inference stage owns the model session, the tracking stage owns the tracker. No
  resource in the frame path is shared between threads or guarded by a lock. The
  queues are the only cross-thread channel.
- Frames flow in decode order from end to end. Because each stage is single-threaded
  and the queues are first in, first out, ordering is a consequence of the structure
  rather than something restored after the fact.

### Backpressure

- Queues are bounded and blocking. A stage that cannot hand off its output waits.
  Frames are never dropped, skipped, or coalesced.
- A saturated downstream stage stalls the decoder. This is correct behavior, not a
  fault, and is neither worked around nor reported as an error.

### Termination

- End of input terminates cleanly: the decoder stops producing, and the shutdown
  cascades downstream so that every frame already in flight is fully processed,
  tracked, and rendered before the run ends. No in-flight frame is discarded at
  normal termination.
- A quit key at the renderer terminates the run promptly, propagating upstream. A
  stage blocked on a full or empty queue must observe the request rather than waiting
  for a frame that will never come.
- The sustained-run duration limit terminates the run the same way end of input does.
- Every spawned thread is joined before the process exits. No thread is detached,
  abandoned, or left running past the run.
- Termination is bounded in time in every case. A run that cannot make progress must
  end with an error rather than hang.

### Error propagation

- A failure in any stage terminates the whole pipeline. The failing stage stops, the
  shutdown request propagates in both directions, and the remaining stages wind down.
- The reported error is the first failure that occurred. Subsequent failures caused by
  the resulting shutdown are logged, not promoted, and never displace the original
  cause.
- A panicking stage thread becomes a reported error with the stage named. It never
  becomes a hang or a silent success.
- Failures carry the stage name and the frame index, continuing VE-011's error
  context discipline.
- Display window cleanup and exit-code behavior established in earlier milestones are
  preserved, including the existing rule that a cleanup failure is reported only when
  nothing else failed.
- Conditions that are normal operating states remain non-fatal: a frame with no
  detections, a frame with no live tracks, a rejected filter update, and a source with
  no usable media timestamps.

### Behavioral parity

- The track dump produced by a threaded run over the VE-012 baseline input is
  byte-identical to the committed serial single-pass baseline dump.
- Parity is asserted, not inspected. A difference fails this pair.
- If parity cannot be achieved, the pair is not complete. Adjusting the baseline to
  match the new output is not an acceptable resolution.

## Constraints and non-goals

- No frame dropping under load. That belongs with live camera support.
- No worker pools, reorder buffers, thread affinity, or priority tuning.
- No buffer pool or allocation optimization, even if per-frame allocation is observed
  to cost throughput. It is measured and reported in VE-016 and addressed later.
- No new instrumentation output. VE-015 adds that.
- No change to detection, tracking, or rendering logic.

## Acceptance criteria

1. Four stage threads run concurrently with rendering on the main thread, and the
   pipeline is demonstrably overlapping work rather than executing serially.
2. Each stage owns its resources exclusively, with no lock in the frame path.
3. Frames are rendered in decode order with none lost, duplicated, or reordered.
4. End of input drains the pipeline: the number of frames rendered equals the number
   of frames decoded.
5. The quit key ends the run promptly, including while a stage is blocked on a queue,
   and all threads are joined.
6. An induced failure in each stage terminates the run, returns non-zero, and reports
   that stage's error as the cause, with the frame index included.
7. An induced panic in a stage thread is reported as an error naming the stage, and
   does not hang the run.
8. The threaded track dump is byte-identical to the VE-012 serial baseline dump.
9. An automated test runs frames through the full threaded pipeline and terminates
   without a timeout backstop being what ends it.
10. Formatting, linting, tests, and the release build pass, and existing tracking
    acceptance tests pass unchanged.
