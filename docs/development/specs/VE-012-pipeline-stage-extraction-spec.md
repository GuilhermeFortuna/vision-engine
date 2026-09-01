# VE-012: Pipeline stage extraction and serial baseline

**Status:** `READY` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-011  
**Implementation plan:** [`../plans/VE-012-pipeline-stage-extraction-plan.md`](../plans/VE-012-pipeline-stage-extraction-plan.md)

## Purpose

Give Milestone 3 its seams and its reference point. The runtime is one function
that decodes, preprocesses, infers, tracks, and renders inline, so there is nothing
for a pipeline to be built out of. This pair separates those five responsibilities
into modules while execution stays strictly serial, then records the resulting
behavior and throughput as the baseline that every later pair in the batch is
measured against.

Separating the restructuring from the concurrency is the point. When identities or
throughput change later in the batch, this boundary is what makes it possible to say
which change caused it.

## Requirements

### Stage separation

- Decode, preprocess, inference, tracking, and rendering each live in their own
  module under a pipeline module. Each stage exposes a function that takes the
  previous stage's output and returns its own.
- Argument parsing, startup validation, and configuration move out of the executable
  entry point into their own module. The entry point retains only wiring: parse,
  validate, run, report.
- The detection module retains model loading and model contract validation. Frame
  preprocessing and output postprocessing move to the preprocess and inference
  stages respectively.
- Postprocessing, including class-aware suppression and letterbox inversion, belongs
  to the inference stage. It is not a sixth stage.
- Tests move with the code they cover.

### Behavior preservation

- Execution remains serial and single-threaded. The stages are called in sequence in
  one loop.
- Observable behavior is unchanged: identical detections, identical identities,
  identical rendering, identical logging, identical exit codes, identical error
  messages, and unchanged command-line surface apart from the addition below.
- No stage trait, no generic pipeline abstraction, and no queue is introduced by this
  pair. The stage functions are ordinary functions over concrete types.
- The project remains a single crate.

### Track dump and deterministic frame bound

- A command-line option writes the per-frame track stream to a file. It is disabled
  by default and changes nothing when absent.
- A second command-line option bounds a run to a fixed number of processed frames,
  continuing across loop boundaries when sustained looping is active. It is disabled
  by default and changes nothing when absent.
- The frame bound exists because the existing duration limit is wall-clock based and
  therefore yields a different frame count on every run. A dump that is to be
  compared byte for byte needs a run length that does not depend on machine speed.
  Without it there is no reproducible baseline for the rest of the batch.
- Each record identifies the frame index, the track identity, the class, the
  bounding box, and the track state, in a stable, diffable, line-oriented text form
  with fixed numeric formatting.
- Records appear in frame order, and within a frame in a deterministic order that
  does not depend on iteration order of any unordered collection.
- The dump reflects the tracks as the renderer receives them, so it describes what
  was displayed rather than an independently recomputed result.

### Serial baseline

- With the extraction complete, record a baseline for a designated local sample
  video and model, committed to the repository:
  - the full track dump for one complete pass over the input,
  - the full track dump for a frame-bounded run long enough to cross at least one
    loop back to the start of the input,
  - a throughput figure from repeated sustained runs, reported as the individual run
    values and their median, not a single measurement.
- The baseline records the sample video, the model, the command line, the commit, and
  the machine's core count, so a later comparison can state whether it is comparing
  like with like.
- The baseline is data, not a test fixture with a passing threshold. This pair adds
  no assertion against it; VE-014 and VE-016 consume it.

## Constraints and non-goals

- No threads, no channels, no queues, no backpressure.
- No performance optimization, no buffer pooling, no allocation strategy changes.
- No behavioral change to detection, tracking, or rendering logic.
- No multi-crate workspace.
- No new runtime dependencies.
- No configuration surface beyond the track dump and frame bound options.

## Acceptance criteria

1. The five stages exist as separate modules, each with a function over concrete
   input and output types, and the executable entry point contains no decoding,
   inference, tracking, or drawing code.
2. Command-line parsing and validation live outside the entry point, and the entry
   point is materially smaller than before this pair.
3. The detection module no longer contains preprocessing or postprocessing.
4. Running the release binary on the sample video produces the same visible result,
   the same logs, and the same exit codes as the previous commit.
5. The track dump option produces a deterministic, frame-ordered record, and two runs
   over the same input produce byte-identical files.
6. The frame bound stops a run at exactly the requested number of processed frames,
   including when the run has looped back to the start of the input.
7. Omitting either option leaves behavior unchanged.
8. The committed baseline contains both track dumps, the individual and median
   throughput figures, and the environment details listed above.
9. Formatting, linting, tests, and the release build pass, and the existing tracking
   acceptance tests pass unchanged.
