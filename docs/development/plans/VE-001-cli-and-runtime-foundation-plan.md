# VE-001 implementation plan: CLI and runtime foundation

**Status:** `READY` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-001-cli-and-runtime-foundation-spec.md`](../specs/VE-001-cli-and-runtime-foundation-spec.md)  
**Depends on:** None

## Current-system context

The repository is a single Rust package. `src/main.rs` only prints a greeting;
`anyhow`, OpenCV, `tracing`, and `tracing-subscriber` are already declared. There
is no command parser, application boundary, model, sample video, or existing
module structure to preserve.

## Implementation decisions

- Keep the implementation in `src/main.rs`. One configuration struct and small
  parsing/validation functions do not justify a module split.
- Parse `std::env::args_os()` directly so paths remain `OsString`/`PathBuf` and a
  new CLI dependency is unnecessary.
- Treat `-h` and `--help` as a successful control result distinct from validated
  runtime configuration. Print usage to stdout for help and stderr for errors.
- Default the model path to `models/yolov8n.onnx`. Validate both paths with
  filesystem metadata and distinguish missing paths from non-regular files.
- Use `anyhow::Result` at the application boundary and add role-specific context
  around filesystem failures. `main` prints the error chain and exits with code 1.
- Initialize `tracing-subscriber` before running the application, with an `info`
  default and no additional logging configuration surface.

## Ordered implementation

1. Replace the greeting with `main` plus a fallible `run` function.
2. Add the configuration and argument-parse result types.
3. Implement OS-native parsing for the required video and optional model path,
   including help and invalid-syntax handling.
4. Validate that both selected paths exist and are regular files.
5. Initialize logging and emit one startup record after validation.
6. Add focused tests for parsing and path validation using temporary files and
   directories created with the standard library.

## Validation

- Test required, explicit-model, default-model, help, unknown-option, missing-value,
  and extra-positional cases.
- Test missing paths and directories used where regular files are required.
- Verify errors identify `video` or `model` and the affected path.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
cargo run -- --help
```

## Handoff

Report the final command syntax, parsing tests, and release-build result. Note that
successful startup only validates paths in this pair; VE-002 first opens the video
and VE-003 first loads the model.
