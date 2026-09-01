# [AGENTS.md](http://AGENTS.md)

## Project

**Vision Engine** is a high-performance, local-first visual intelligence platform written primarily in Rust.

Read `PROJECT.md` before making architectural or scope decisions. Treat `PROJECT.md`, task-specific specs, plans, and the current task prompt as the sources of truth for milestones, checkpoints, sequencing, and current implementation scope.

---

## Agent Operating Principles

- Keep changes small, bounded, visible, and testable.
- Prefer the simplest correct implementation.
- Do not introduce abstractions before they are needed.
- Avoid speculative architecture.
- Do not silently expand scope beyond the requested task.
- Measure performance changes instead of assuming they help.
- Inspect existing code before modifying it.
- Preserve unrelated user changes.

---



## Agent Workflow

For every implementation task:

1. Read:
  - `AGENTS.md`
  - `PROJECT.md`
  - any task-specific spec or plan
2. Inspect the existing code before modifying it.
3. Determine the exact task boundary from the current task/spec.
4. Implement only what is necessary for that task.
5. Run all relevant validation commands.
6. Fix validation failures caused by the change.
7. Review the diff before finishing.
8. Report:
  - what changed
  - validation performed
  - any known limitation or follow-up

Do not infer milestone progression or begin follow-up work unless explicitly requested.

---

## Semantic Code Search

Use the local semantic code-search MCP for **conceptual and architecture** questions before broad repository exploration.

Tools:

- `semantic_code_search` — hybrid retrieval over Rust code and development docs
- `task_context` — VE spec/plan sections plus related code for a task id (use `ve_id`, e.g. `VE-014`; `task_id` is an alias)
- `reindex` — force a full index rebuild after large refactors

Default behavior:

- For architecture, stage boundaries, or “where does X happen?” questions, call `semantic_code_search` first with `top_k=8`.
- For numbered `VE-...` tasks, call `task_context(ve_id="VE-013")` before reading specs manually.
- During implementation, use `path_prefix` (e.g. `src/pipeline/`) to reduce noise in search results.
- When you already know the symbol, path, or filename, use grep or direct file reads instead.
- Verify hits using `line_start` / `line_end` and inspect call sites before treating results as authoritative.

Search tools return a wrapped response with `results` and `index` metadata. Check `index.reindexed` and `index.files_changed_since_index` during active implementation — the index auto-rebuilds when tracked files change, but grep/read remain authoritative once symbols are known.

Do not treat MCP results as authoritative by themselves. Verify surrounding code when correctness depends on call sites, types, configuration, or cross-function behavior.

Do not use semantic search for:

- exact filename or path lookups
- exact symbol searches when the symbol name is already known

The tool indexes production Rust items under `src/**/*.rs` (functions, structs, enums, impls), VE specs/plans under `docs/development/`, plus `AGENTS.md` and `PROJECT.md`. Test-only code is excluded. Embeddings use the local Ollama model `qwen3-embedding:4b`.

---

## VE Tasks

When working on a numbered Vision Engine task (`VE-...`):

1. Identify the task from its spec or plan under `docs/development/specs/` or `docs/development/plans/`.
2. Create and use a dedicated git branch named after that document (filename without `.md`).
   - Example: for `docs/development/specs/VE-006-tracking-domain-and-frame-timestamps-spec.md`, use branch `VE-006-tracking-domain-and-frame-timestamps-spec`.
3. Do all implementation work for that task on that branch.
4. Do not mix changes from different `VE-...` tasks on the same branch.

Create the branch before making changes if it does not already exist.

---



## Primary Stack



### Language

- Rust
- Cargo



### Runtime libraries

Use project dependencies intentionally and only when justified by the task.

Common project libraries may include:

- `anyhow`
- `tracing`
- `tracing-subscriber`



### Computer vision

OpenCV is the primary computer-vision library for functionality such as:

- video capture
- decoding
- image manipulation
- rendering
- display

Other inference, concurrency, storage, or GPU dependencies should only be introduced when required by the current task or project plan.

---



## Repository Structure

Keep the repository simple unless real complexity justifies restructuring.

Do not convert the project into a multi-crate workspace prematurely.

Avoid creating modules, traits, services, factories, interfaces, queues, or generic abstractions for code that currently has only one implementation.

---



## Rust Commands

Unless a task says otherwise, run these commands from the **repository root**.

### Fast compile check

```bash
cargo check
```



### Formatting

```bash
cargo fmt --check
```

If formatting fails:

```bash
cargo fmt
```



### Linting

```bash
cargo clippy --all-targets --all-features -- -D warnings
```



### Tests

```bash
cargo test
```



### Release build

```bash
cargo build --release
```

For changes affecting the executable path, a successful release build is required before completion.

---



## Validation Requirements

Before considering an implementation task complete, run at minimum:

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo build --release
```

If any command cannot be run because of an environment or external dependency issue:

- do not hide the failure
- report the exact blocker
- distinguish environment failures from code failures

---



## Error Handling

- Prefer `anyhow::Result` at application boundaries.
- Return useful errors rather than panicking.
- Do not use `unwrap()` or `expect()` in normal runtime paths unless an invariant is truly guaranteed and the reason is obvious.
- Include enough context for failures to be actionable.

---



## Logging and Metrics

Use `tracing` for diagnostic output when appropriate.

Prefer measurable values over vague performance claims.

Do not introduce complex telemetry unless the task requires it.

---



## Performance Guidance

Performance matters, but correctness comes first.

- Avoid unnecessary copies when reasonably simple.
- Do not add unsafe code for performance without evidence.
- Do not introduce concurrency merely because it may be useful later.
- Benchmark meaningful optimizations.
- Prefer a correct measurable baseline before optimization.

---



## Dependency Policy

- Prefer well-maintained crates with clear justification.
- Avoid adding dependencies for functionality easily handled by the standard library.
- Do not add future dependencies ahead of the work that needs them.
- Keep dependency versions intentional.
- Explain non-obvious dependency additions in the task summary.

---



## Code Quality

Prefer:

- idiomatic Rust
- small functions
- explicit ownership
- clear error propagation
- simple control flow
- descriptive names
- minimal mutable state

Avoid:

- unnecessary cloning
- premature generics
- excessive trait abstraction
- large dependency trees without need
- hidden global state
- unsafe code without strong justification

---



## Git Discipline

For `VE-...` tasks, use one branch per task named after the task document (see **VE Tasks** above).

Keep commits focused on one logical change.

Do not mix unrelated:

- refactors
- dependency upgrades
- formatting
- cleanup

with a feature unless required for that feature.

Do not rewrite unrelated user changes.

---



## Scope Authority

`AGENTS.md` defines durable repository-wide working rules for agents.

It must **not** define:

- the current checkpoint
- milestone progression
- task ordering
- the next implementation task
- temporary scope boundaries
- checkpoint-specific definitions of done

Those belong in `PROJECT.md`, task-specific specs/plans, or the current task prompt.