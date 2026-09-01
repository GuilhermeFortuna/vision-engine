# VE-015: Pipeline instrumentation

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-014  
**Implementation plan:** [`../plans/VE-015-pipeline-instrumentation-plan.md`](../plans/VE-015-pipeline-instrumentation-plan.md)

## Purpose

Make the pipeline's behavior legible. A concurrent pipeline hides where time goes:
the visible frame rate is set by the slowest stage, and nothing on screen says which
one that is. This pair reports per-stage latency and queue depth live, at end of run,
and in the sustained-run record, so Milestone 3's acceptance rests on numbers and so
later optimization work starts from measurement rather than guesswork.

## Requirements

### Per-stage latency

- Each stage measures the time it spends on each frame: decode, preprocess,
  inference, tracking, and rendering.
- Measurements travel with the frame, so every reported value describes the frame it
  is attached to. No value is averaged across frames before display unless it is
  labelled as an average, and no value is carried over from a previous frame without
  being marked unavailable.
- The inference measurement continues to report the model execution time already
  measured, rather than re-timing or re-deriving it.

### Queue depth

- The depth of each of the four queues is reported alongside the stage latencies.
- Depth is sampled, not accumulated per frame. A sampled depth is an instantaneous
  observation and is presented as one.
- Depth is reported in a form that makes saturation visible, so a queue sitting at
  capacity can be distinguished from one sitting empty without arithmetic by the
  reader.

### Bottleneck reporting

- The end-of-run summary names the stage with the highest mean latency and reports
  the observed saturation pattern of the queues, so the bottleneck is stated rather
  than left to be inferred.
- The summary reports, per stage, the mean and a high-percentile latency, and per
  queue, the mean depth and the fraction of samples at capacity.
- The summary retains the frame count, media time, timestamp provenance, adjustment
  count, and rejected-update figures reported since VE-011.

### Live overlay

- The overlay reports the five stage latencies, the four queue depths, end-to-end
  frames per second, and the confirmed track count, all readable at once.
- The overlay area grows to fit, and label placement continues to keep track labels
  clear of it, preserving the behavior established in VE-010.
- Frames per second at the renderer is throughput, not decode rate, and the overlay
  presents it as such. Under backpressure the decoder is deliberately idle, and the
  displayed rate must not be readable as a decoder measurement.

### Sustained-run record

- The sustained-run script's per-sample record gains the per-stage latencies and the
  queue depths, keeping its existing sampling cadence, warm-up, and tolerance.
- Existing columns are retained so earlier records remain comparable.

### Cost

- Instrumentation must not measurably reduce throughput. The cost is measured against
  VE-014's figures and reported. If it is not negligible, the measurement is reduced
  in scope rather than left in place unreported.

## Constraints and non-goals

- No microbenchmark suite, no benchmarking framework, no separate profiling harness.
  Milestone 3 measures the real end-to-end pipeline.
- No trace export, no metrics endpoint, no time-series storage, no external
  monitoring integration.
- No configurable metrics, no verbosity levels, no overlay toggles or themes.
- No optimization work in response to what the numbers show. This pair reports; it
  does not tune.
- No change to pipeline structure, stage behavior, or tracking logic.

## Acceptance criteria

1. Five per-stage latencies, four queue depths, end-to-end frames per second, and the
   confirmed track count appear on the overlay simultaneously and legibly.
2. Every displayed value describes measured work on the frame being displayed, or is
   explicitly labelled as an average or as unavailable.
3. Track labels remain clear of the enlarged overlay at every frame edge.
4. The end-of-run summary reports per-stage mean and high-percentile latency, per
   queue mean depth and time at capacity, and names the slowest stage.
5. The summary retains all figures reported since VE-011.
6. The sustained-run record contains the new columns alongside the existing ones, and
   its sampling behavior is unchanged.
7. A saturated queue and a starved queue are distinguishable from the reported
   figures alone, demonstrated on a real run.
8. Throughput with instrumentation enabled is compared against VE-014 and the
   difference is reported.
9. Value formatting and summary derivation are unit tested without requiring a
   display.
10. Formatting, linting, tests, and the release build pass.
