# VE-016: Milestone 3 hardening and acceptance

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-015  
**Implementation plan:** [`../plans/VE-016-milestone-three-hardening-and-acceptance-plan.md`](../plans/VE-016-milestone-three-hardening-and-acceptance-plan.md)

## Purpose

Turn the working pipeline into the completed Milestone 3 baseline. This pair proves
three things with recorded numbers rather than impressions: the concurrent pipeline
produces exactly the identities the serial one did, it is measurably faster, and it
cannot deadlock or grow without bound. Concurrency failures are rare and
non-deterministic by nature, so the checks here are designed to provoke them rather
than to wait for them.

## Requirements

### Behavioral parity

- The track dump from the full pipeline is byte-identical to the VE-012 serial
  baseline for the baseline input, re-verified here rather than assumed from VE-014.
- Parity is verified over a longer run than VE-014's, using the frame bound from
  VE-012 so the run length is reproducible, and long enough to cross at least one
  loop back to the start of the input, so behavior is confirmed beyond the first
  pass.
- Parity holds across repeated runs. Non-determinism between two threaded runs is a
  defect, not a tolerance.
- If parity fails, the cause is identified and fixed. The baseline is not revised to
  match, and no assertion is relaxed to pass.

### Throughput

- Throughput is measured on the same input, model, and machine as the VE-012
  baseline, and the comparison states the commit and core count of both.
- Multiple runs are taken for each configuration. Individual values and the median
  are reported. A single measurement does not satisfy this requirement.
- The result is reported as a ratio against the serial baseline, together with the
  per-stage latencies and queue depths that explain it.
- An improvement is required for acceptance. If the pipeline is not faster, that is
  reported as a measured outcome with the bottleneck named from the instrumentation,
  and it blocks milestone acceptance rather than being explained away.
- Per-frame allocation cost introduced by giving each frame its own buffer is
  quantified and reported, since it is the known way this refactor could lose
  throughput.

### Liveness and boundedness

- A sustained run holds for a stated duration without deadlock, livelock, stall, or
  unbounded growth. Resident memory and queue depths are sampled throughout and
  reported as a series, not a final value.
- Every queue stays within its capacity for the whole run. Depth never grows without
  bound, by construction and by observation.
- Deliberately stalling each stage in turn is exercised as a test: the pipeline
  applies backpressure, does not consume unbounded memory, and still shuts down
  cleanly when asked. A stalled pipeline must remain a stopping pipeline.
- Shutdown is exercised from each state that could hang it: while queues are full,
  while queues are empty, while a stage is mid-frame, at the moment of end of input,
  and immediately at startup before the first frame. Each terminates and joins every
  thread.
- Shutdown while blocked is verified without relying on a timeout as the mechanism
  that ends the test.
- Repeated start and stop cycles within one process leave no thread, resource, or
  memory growth behind.

### Failure behavior

- Milestone 1 and 2 startup and lifecycle error behavior is preserved unchanged.
- Runtime failures return non-zero. Help, end of input, and the quit keys return
  success. Expected failures do not panic and do not print a backtrace unless the
  user has enabled standard backtrace behavior.
- Every stage's induced failure and induced panic terminates the run with that stage
  named as the cause, re-verified here across the whole set.

### Measured acceptance run

- Build and run the release executable with a local YOLOv8n COCO ONNX model and a
  representative local video containing recognizable, moving COCO objects.
- Visually confirm that boxes track their objects, identities persist as objects
  move, and the full instrumentation overlay is readable during the run.
- Record and report: the parity result, the throughput ratio with individual and
  median values, the per-stage latency breakdown, the queue saturation pattern with
  the bottleneck named, the memory series, and the identity churn figure from VE-011
  re-measured on the pipeline.
- Milestone 3 is accepted only when parity holds, throughput has improved, no
  liveness check failed, and all validation passes. The status document is updated to
  record the milestone as complete, with the measurements referenced.

## Constraints and non-goals

- No optimization beyond what is needed to pass the throughput requirement, and any
  optimization made is measured before and after.
- No new features, stages, or configuration.
- No worker pools, buffer pools, or drop policies. These are candidates for later
  milestones, recorded as follow-ups with the measurements that would justify them.
- No relaxation of any earlier milestone's acceptance criteria.

## Acceptance criteria

1. The track dump matches both committed VE-012 baselines byte for byte, including
   the frame-bounded run that crosses a loop of the input, and repeated runs agree
   with each other.
2. Throughput is reported as individual and median values for both configurations,
   with commit and core count stated, and shows an improvement over the serial
   baseline.
3. The per-stage latency and queue saturation figures are reported and the bottleneck
   stage is named.
4. Per-frame allocation cost is quantified and reported.
5. A sustained run of the stated duration completes with no deadlock or stall, with
   memory and queue-depth series recorded and no unbounded growth.
6. Every queue is observed within capacity for the entire run.
7. Stalling each stage in turn produces backpressure, bounded memory, and a clean
   shutdown on request.
8. Shutdown succeeds from full queues, empty queues, mid-frame, at end of input, and
   at startup, joining every thread each time, without a timeout being the mechanism
   that ends the test.
9. Repeated start and stop cycles leave no residual threads or growth.
10. Induced failure and induced panic in every stage return non-zero and name the
    stage.
11. Exit-code and error-reporting behavior from Milestones 1 and 2 is unchanged.
12. Formatting, linting, tests, and the release build pass, and the status document
    records Milestone 3 as accepted with its measurements.
