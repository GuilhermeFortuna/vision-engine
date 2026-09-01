# VE-013 implementation plan: Frame messages and bounded queues

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Specification:** [`../specs/VE-013-frame-messages-and-bounded-queues-spec.md`](../specs/VE-013-frame-messages-and-bounded-queues-spec.md)  
**Depends on:** VE-012

## Current-system context

After VE-012 the five stages are separate functions and the serial loop in
`pipeline::run` carries the frame, stamp, prepared tensor, detections, and tracks as
locals. A queue carries one owned value per hop, so those locals must be bundled. That
bundling, and the queue itself, are all this pair builds.

Two facts drive the design. First, `DecodeStage::next_into` writes into a reused
`Mat`; once frames travel through a queue each frame needs its own, so the decode
signature changes here. Second, OpenCV's `Mat` is reference-counted internally, and
whether it can be moved between threads must be settled before VE-014 depends on it.

## Interfaces produced

```rust
// src/pipeline/message.rs
pub struct StageTimings {
    pub decode_ms: f64,
    pub preprocess_ms: f64,
    pub inference_ms: f64,
    pub tracking_ms: f64,
}

pub struct DecodedFrame {
    pub frame: Mat,
    pub stamp: FrameStamp,
    pub timings: StageTimings,
}
pub struct PreparedFrame {
    pub frame: Mat,
    pub stamp: FrameStamp,
    pub timings: StageTimings,
    pub input: Array4<f32>,
    pub transform: LetterboxTransform,
}
pub struct DetectedFrame {
    pub frame: Mat,
    pub stamp: FrameStamp,
    pub timings: StageTimings,
    pub detections: Vec<Detection>,
}
pub struct TrackedFrame {
    pub frame: Mat,
    pub stamp: FrameStamp,
    pub timings: StageTimings,
    pub tracks: Vec<Track>,
}

// src/pipeline/queue.rs
pub struct Shutdown(Arc<AtomicBool>);
impl Shutdown {
    pub fn new() -> Self;
    pub fn request(&self);
    pub fn is_requested(&self) -> bool;
    pub fn clone_handle(&self) -> Self;
}

pub struct Sender<T>;
pub struct Receiver<T>;
pub struct Disconnected;

pub fn bounded<T>(capacity: usize, shutdown: &Shutdown) -> (Sender<T>, Receiver<T>);

impl<T> Sender<T> {
    pub fn send(&self, item: T) -> Result<(), Disconnected>;
}
impl<T> Receiver<T> {
    pub fn recv(&self) -> Result<T, Disconnected>;
    pub fn len(&self) -> usize;
    pub fn capacity(&self) -> usize;
}

pub const QUEUE_CAPACITY: usize = 2;
```

Stage signatures change to consume and produce messages:

```rust
impl DecodeStage { pub fn next(&mut self) -> Result<Option<DecodedFrame>>; }
pub fn prepare(decoded: DecodedFrame) -> Result<PreparedFrame>;
impl InferStage { pub fn detect(&mut self, prepared: PreparedFrame) -> Result<DetectedFrame>; }
impl TrackStage { pub fn update(&mut self, detected: DetectedFrame) -> Result<TrackedFrame>; }
impl RenderStage { pub fn present(&mut self, tracked: TrackedFrame,
    metrics: &FrameMetrics) -> Result<Presentation>; }
```

## Implementation decisions

- The queue is written rather than taken from `crossbeam-channel`. It needs three
  things a general channel does not give together: an observable depth for VE-015's
  metric, a shutdown signal that wakes blocked ends, and semantics we can state
  exactly in tests. It is a `Mutex<VecDeque<T>>` with a not-empty and a not-full
  condition variable, on the order of a hundred lines. The dependency would still
  need a shutdown mechanism layered on top of it.
- One `Shutdown` handle is shared by all four queues, so a single request wakes every
  blocked stage. It is an `Arc<AtomicBool>`; a `Condvar` broadcast follows the flag so
  that setting it wakes waiters rather than leaving them until their next wakeup.
- Blocked ends re-check the shutdown flag under the mutex after every wait, never on a
  timer. A test that passes only because a wait timed out is not evidence of anything,
  which is why the spec forbids relying on one.
- `send` on a full queue blocks; on a disconnected receiver or a requested shutdown it
  returns `Disconnected` and the item is dropped. `recv` on an empty queue blocks; when
  all senders are gone it drains remaining items first and only then reports
  `Disconnected`. Draining before disconnecting is what makes VE-014's end-of-input
  path lose no in-flight frames.
- `Disconnected` is a unit struct, not an error type. Disconnection is a normal
  termination signal, and making it an `anyhow::Error` would invite stages to report a
  clean shutdown as a failure.
- Capacity is 2. It is enough to overlap one frame per stage while keeping in-flight
  frames, and therefore memory, to a handful of full-resolution images. Larger buffers
  only hide which stage is slow, which is the opposite of what VE-015 needs.
- `StageTimings` accumulates on the message so the renderer reports the timings of the
  frame it is drawing. With four stages in flight simultaneously, per-stage timings
  held in the renderer would mix four different frames and be quietly wrong.
- Each message owns its `Mat`. `DecodeStage::next` allocates a fresh `Mat` per frame
  instead of reusing one. This is a real per-frame cost and it is accepted here, not
  optimized: VE-016 quantifies it and a buffer pool is a later milestone's work.
- `Send` is proven with a compile-time assertion function, not a runtime test, so a
  regression is a build failure. If any message type is not `Send`, stop and implement
  the stated fallback: convert to an owned `Vec<u8>` plus dimensions at the decode
  boundary, and record the per-frame copy cost in the handoff.
- The binary still runs serially at the end of this pair. `pipeline::run` constructs
  each message and passes it to the next stage directly. No queue is used in the
  binary yet; queues are exercised by tests only. This keeps the concurrency change
  isolated to VE-014.

## Ordered implementation

1. Create the branch `VE-013-frame-messages-and-bounded-queues-spec`.
2. Create `src/pipeline/message.rs` with `StageTimings` and the four message types.
3. Add the compile-time send assertion and run the build:

```rust
fn assert_send<T: Send>() {}

#[test]
fn messages_move_between_threads() {
    assert_send::<DecodedFrame>();
    assert_send::<PreparedFrame>();
    assert_send::<DetectedFrame>();
    assert_send::<TrackedFrame>();
}
```

4. Run `cargo test messages_move_between_threads`. If it does not compile, stop and
   implement the owned-buffer fallback described above before continuing.
5. Change the stage signatures to consume and produce messages, and rewrite
   `pipeline::run` to thread the messages through serially. `DecodeStage::next`
   allocates its own `Mat` and returns `Option<DecodedFrame>`. Run the full suite and
   confirm the track dump still matches the VE-012 baseline. Commit.
6. Write the first failing queue test, order and blocking on empty:

```rust
#[test]
fn recv_blocks_until_an_item_arrives_and_preserves_order() {
    let shutdown = Shutdown::new();
    let (tx, rx) = bounded::<u32>(2, &shutdown);
    let producer = thread::spawn(move || {
        for value in 0..3 { tx.send(value).unwrap(); }
    });
    assert_eq!(rx.recv().unwrap(), 0);
    assert_eq!(rx.recv().unwrap(), 1);
    assert_eq!(rx.recv().unwrap(), 2);
    producer.join().unwrap();
}
```

7. Run it and confirm it fails to compile because `queue` does not exist.
8. Implement `src/pipeline/queue.rs`: `Shutdown`, the shared inner state
   (`Mutex<VecDeque<T>>`, `Condvar` for not-empty and for not-full, sender and receiver
   counts), `bounded`, `send`, `recv`, `len`, `capacity`, and the `Drop`
   implementations that decrement the counts and notify both condition variables.
9. Run the test and confirm it passes. Commit.
10. Write a failing test that `send` blocks when full:

```rust
#[test]
fn send_blocks_while_the_queue_is_full() {
    let shutdown = Shutdown::new();
    let (tx, rx) = bounded::<u32>(2, &shutdown);
    let sent = Arc::new(AtomicUsize::new(0));
    let counter = Arc::clone(&sent);
    let producer = thread::spawn(move || {
        for value in 0..4 { tx.send(value).unwrap(); counter.fetch_add(1, SeqCst); }
    });
    // Two items fit; the third send cannot complete until a receive frees space.
    while sent.load(SeqCst) < 2 { std::hint::spin_loop(); }
    assert!(sent.load(SeqCst) <= 3, "sender ran past the queue capacity");
    for expected in 0..4 { assert_eq!(rx.recv().unwrap(), expected); }
    producer.join().unwrap();
}
```

11. Run it and confirm it passes with the implementation from step 8, or fix the
    implementation until it does. Commit.
12. Write failing tests for disconnection: dropping the sender lets the receiver drain
    two queued items and then returns `Disconnected`; dropping the receiver makes a
    blocked sender return `Disconnected`. Implement until both pass. Commit.
13. Write a failing test that shutdown wakes both ends:

```rust
#[test]
fn shutdown_wakes_a_blocked_receiver() {
    let shutdown = Shutdown::new();
    let (_tx, rx) = bounded::<u32>(2, &shutdown);
    let waiter = thread::spawn(move || rx.recv());
    shutdown.request();
    assert!(waiter.join().unwrap().is_err());
}
```

14. Add the matching test for a sender blocked on a full queue. Implement the
    condition-variable broadcast in `Shutdown::request` until both pass. Commit.
15. Write a failing throughput test: one producer sends ten thousand sequential
    integers through a capacity-2 queue while one consumer receives them, asserting
    every value arrives exactly once in order and the count matches. Make it pass.
    Commit.
16. Write a failing test for `len`: after three sends and one receive on a
    capacity-4 queue, `len` is 2, and `capacity` is 4. Make it pass. Commit.
17. Run the full validation suite and confirm the binary's behavior and track dump are
    unchanged from VE-012.

## Validation

- Unit: order preservation; blocking on empty; blocking on full; sender-drop drains
  then disconnects; receiver-drop disconnects a blocked sender; shutdown wakes both
  ends; ten-thousand-item producer and consumer exchange; `len` and `capacity`.
- Compile-time: all four message types are `Send`.
- Regression: the track dump over the baseline input is byte-identical to the VE-012
  single-pass baseline; existing tests pass unchanged.
- No test may use a sleep or a timeout as the mechanism that makes it pass. Blocking
  is demonstrated by observed progress under a bounded queue, not by waiting.

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test
cargo test --release queue          # race conditions surface under optimization
cargo build --release
target/release/vision-engine samples/test.mp4 --model models/yolov8n.onnx \
  --track-dump /tmp/ve013.csv
diff /tmp/ve013.csv docs/development/baselines/VE-012/single-pass.csv
```

## Handoff

Report whether the message types are `Send` unmodified or whether the owned-buffer
fallback was needed and what it costs per frame, the queue implementation's line count,
the chosen capacity and the reasoning if it changed from 2, confirmation that no test
depends on a sleep or timeout, and confirmation that the track dump still matches the
VE-012 baseline byte for byte.
