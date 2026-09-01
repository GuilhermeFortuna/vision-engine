# VE-011: Milestone 2 hardening and acceptance

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-010  
**Implementation plan:** [`../plans/VE-011-milestone-two-hardening-and-acceptance-plan.md`](../plans/VE-011-milestone-two-hardening-and-acceptance-plan.md)

## Purpose

Turn the working tracker into the completed Milestone 2 baseline. This pair closes
failure-path gaps introduced by tracking, and proves identity stability and bounded
resource use with measurements rather than impressions. Every acceptance check here
is defined so that it passes or fails on a recorded number, not on a judgement call.

## Requirements

### Failure behavior

- Milestone 1's startup and lifecycle error behavior is preserved unchanged.
- Tracking-specific failures carry distinct, actionable context naming the stage and
  the frame index: rejected filter updates, association failures, and unusable frame
  timestamps.
- A rejected filter update, a frame with no detections, a frame with no live tracks,
  and a source with no usable media timestamps are all normal operating conditions.
  None terminates the run, and none is reported as an error.
- Runtime failures return non-zero. Help, end of file, and Q, q, or Escape return
  success. Expected failures do not panic and do not print a backtrace unless the
  user has enabled standard backtrace behavior.

### Deterministic identity evidence

- A synthetic detection sequence drives the full tracker with no video, model, or
  display, and asserts the exact identity assigned on every frame.
- Five scenarios are required and gate acceptance, each asserted separately:
  continuous tracking of a moving object, an occlusion gap shorter than the
  retention limit that preserves identity, a gap longer than the retention limit
  that issues a new identity, two objects of different classes overlapping without
  interfering, and a spurious single-frame detection that never reaches
  confirmation.
- These five assertions compare identities by value. A test that only asserts a
  track count, or only that some identity exists, does not satisfy this requirement.

### Diagnostic scenario: same-class crossing

- Two objects of the same class converging, overlapping, and separating is exercised
  as a recorded diagnostic, not as a gating assertion. A tracker with no appearance
  or re-identification features cannot guarantee identity through an ambiguous
  same-class crossing, and requiring it here would either fail correct work or
  invite the test to be weakened until it passed.
- The scenario asserts only what the tracker must hold regardless of the outcome:
  the run completes without error, and the expected number of confirmed tracks is
  present before and after the crossing.
- Whether the two identities survived the crossing is recorded as an observation and
  reported with the acceptance evidence. Both outcomes are acceptable for Milestone
  2. A recorded identity switch is the measurement that would justify appearance
  features or a second-stage association in a later milestone; it is not a defect
  to be fixed inside this pair.

### Measured acceptance run

- Build and run the release executable with a local YOLOv8n COCO ONNX model and a
  representative local video containing recognizable, moving COCO objects.
- Record, for a designated segment in which a stated number of objects is
  continuously present: the number of confirmed identities issued during that
  segment, and the resulting identity churn expressed as identities issued per one
  hundred frames. The expected identity count and the observed count are both
  reported. Excess identities are reported as measured instability, not explained
  away.
- Visually confirm that boxes track their objects, identities persist as objects
  move, and the overlay shows decode latency, inference latency, processing frames
  per second, tracking latency, and confirmed track count together.
- Exercise end of file, interactive exit, missing video, unreadable video, missing
  model, invalid ONNX, unsupported tensor shape, and a source without usable media
  timestamps.

### Sustained resource run

- Run the release build continuously for twelve minutes total using a sufficiently
  long or externally repeated input: a two-minute warm-up followed by a ten-minute
  measured window. The warm-up is excluded from the pass criteria but is sampled and
  reported like every other interval.
- Sample every sixty seconds, beginning at the two-minute mark and continuing to the
  twelve-minute mark inclusive. This yields eleven samples, of which the first is the
  post-warm-up baseline and the remaining ten are the measured window.
- Each sample records: elapsed seconds, resident set size in kilobytes, cumulative
  frame count, live track count, and confirmed track count.
- A run that produces fewer than eleven samples, or whose samples are not sixty
  seconds apart, is incomplete and is reported as a blocker rather than evaluated.
- Resident set size passes when the final sample does not exceed the first
  post-warm-up sample by more than five per cent or ten megabytes, whichever is
  larger, and when the final five samples do not form a strictly increasing
  sequence. Allocator high-water behavior means resident set size is not required to
  return to its starting value; a bounded plateau passes and a continuing rise does
  not.
- Live track count passes when it stays bounded across the run and falls back toward
  zero during segments in which the scene is empty. A live track count that rises
  with frame count is a failure regardless of resident set size.
- Report release hardware, video resolution, model identity, run duration, observed
  decode, inference and tracking latency ranges, processing frames per second, and
  every sampled figure above.

### Milestone boundary

- The executable remains the direct synchronous CPU pipeline: decode, preprocess,
  infer, post-process, track, render.
- Completing this pair makes Milestone 2 eligible to be marked complete. It does not
  authorize Milestone 3 work.

## Constraints and non-goals

- No events, zones, persistence, search, camera or network input, GPU execution,
  output encoding, queues, async runtime, parallel workers, or additional model
  families.
- Do not claim identity stability or leak freedom from ownership, a short run, or a
  visual impression. Use the recorded measurements required above.
- Do not weaken a threshold, shorten a run, or narrow a scenario to make a check
  pass. A check that cannot be run is reported as a blocker.
- Do not commit proprietary or large model or video assets to satisfy validation.

## Acceptance criteria

1. All failure cases return the expected status with contextual messages and no
   panic, and all defined normal conditions run without error.
2. All five gating identity scenarios assert exact identities by value and pass.
   The same-class crossing diagnostic runs, holds its weaker invariants, and has its
   observed outcome recorded either way.
3. The release executable tracks recognizable objects with persistent identities and
   displays all five required metrics simultaneously.
4. End of file and interactive exit release the window and terminate promptly.
5. The acceptance run reports expected and observed identity counts and the measured
   identity churn for the designated segment.
6. The sustained run produces all eleven sixty-second samples across the full twelve
   minutes and satisfies the stated resident set size and live track count criteria.
7. Unit and integration tests cover all deterministic logic and practical startup
   failures without requiring a graphical session or network access.
8. Formatting, strict linting, tests, and the release build all pass.
9. Any unavailable model, video, display, or system runtime is reported as an exact
   environment blocker, and blocked evidence is never represented as a passing check.
10. The final diff contains no functionality assigned to Milestone 3 or later.
