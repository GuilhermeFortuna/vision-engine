# VE-001: CLI and runtime foundation

**Status:** `DONE` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** None  
**Implementation plan:** [`../plans/VE-001-cli-and-runtime-foundation-plan.md`](../plans/VE-001-cli-and-runtime-foundation-plan.md)

## Purpose

Replace the placeholder executable with the smallest useful application boundary
for Milestone 1. This pair establishes how a user selects a local video and model,
how startup failures are reported, and how later video and inference work receives
validated configuration.

## Requirements

### Command surface

- The executable is invoked as `vision-engine <video> [--model <path>]`.
- `<video>` is one required local filesystem path.
- `--model` is optional and defaults to `models/yolov8n.onnx` relative to the
  process working directory.
- `-h` and `--help` print concise usage and exit successfully without validating
  either path.
- Unknown options, a missing video argument, repeated positional arguments, or a
  missing `--model` value print an actionable error and exit unsuccessfully.
- Argument parsing remains small and local. A CLI framework is not required for
  this command surface.

### Validated configuration

- Successful parsing produces one concrete configuration value containing the
  video and model paths.
- Both paths must exist and be regular files. The error identifies which role and
  path failed without panicking.
- File contents are not decoded or interpreted in this pair. OpenCV and ONNX
  validation belong to the pairs that consume those files.
- Paths are kept as filesystem-native values rather than converted lossily to
  UTF-8 during parsing or validation.

### Application boundary

- `main` delegates to a fallible application function and maps success or failure
  to a process exit status.
- Application errors preserve their source and add enough context to identify the
  failed operation.
- `tracing-subscriber` is initialized once with a compact, human-readable default
  suitable for a local command-line process.
- A valid invocation logs the selected paths and exits successfully until VE-002
  adds the video loop.

## Constraints and non-goals

- No video decoding, model loading, inference, rendering, tracking, persistence,
  camera input, GPU support, async runtime, or concurrency.
- No configuration file or environment-variable layer.
- No model or sample asset is downloaded or committed.
- Do not restructure the repository or introduce traits and services for the one
  executable path.

## Acceptance criteria

1. `vision-engine --help` and `vision-engine -h` show the supported syntax and
   return success without requiring local assets.
2. A valid video path and either an explicit or default valid model path produce
   the same validated configuration contract.
3. Missing, non-file, and malformed argument cases return non-zero with messages
   naming the offending argument or path.
4. Non-UTF-8-capable path handling is not prevented by the parser or configuration
   representation.
5. The placeholder greeting is removed and runtime paths contain no `unwrap()` or
   `expect()`.
6. Formatting, linting, tests, and the release build pass.
