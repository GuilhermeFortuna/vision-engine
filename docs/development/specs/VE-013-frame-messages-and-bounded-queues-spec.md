# VE-013: Frame messages and bounded queues

**Status:** `BLOCKED` (authoritative: [`../STATUS.md`](../STATUS.md))  
**Project direction:** [`../../../PROJECT.md`](../../../PROJECT.md)  
**Depends on:** VE-012  
**Implementation plan:** [`../plans/VE-013-frame-messages-and-bounded-queues-plan.md`](../plans/VE-013-frame-messages-and-bounded-queues-plan.md)

## Purpose

Build the two things the threaded runtime needs and nothing else: the concrete
messages that cross each stage boundary, and a bounded queue that blocks. Both are
built and tested before any thread exists, so when VE-014 introduces concurrency the
transport underneath it is already known to be correct.

The scope is deliberately concrete. The messages are named for the specific edges in
this pipeline, and the queue is one type used four times.

## Requirements

### Frame messages

- One message type per stage boundary, each carrying exactly what the next stage
  needs: the decoded frame, the prepared frame, the detected frame, and the tracked
  frame.
- Every message carries the frame stamp introduced in VE-006, so frame index and
  media time stay attached to the frame for its whole journey.
- Every message carries the decoded image, because the renderer needs the original
  pixels at the end of the pipeline.
- Every message carries the stage timings accumulated so far, so the renderer reports
  the timings of the frame it is drawing rather than a mixture of frames occupying
  different pipeline positions.
- Message ownership is exclusive. A message is moved from stage to stage and is never
  shared, aliased, or referenced by a stage that has passed it on.
- No stage trait, no generic message envelope, no pipeline framework. Messages are
  plain data types.

### Thread-safety of the decoded image

- Whether the decoded image type can be moved between threads is established by a
  test that fails to compile if it cannot, not by assumption.
- If it cannot be moved, the fallback is stated and implemented here: convert to an
  owned buffer at the decode boundary and document the per-frame copy as a known
  cost. Discovering this during VE-014 is a failure of this pair.

### Bounded queue

- One bounded queue type with a sender and a receiver, created with a fixed capacity.
- Sending to a full queue blocks the sender until capacity is available. Receiving
  from an empty queue blocks the receiver until an item arrives. Items are never
  dropped, overwritten, or reordered.
- Delivery is first in, first out, and no item is lost or duplicated under concurrent
  sending and receiving.
- When every sender is gone, a blocked or subsequent receive drains any remaining
  items and then reports disconnection rather than blocking forever. When the
  receiver is gone, a blocked or subsequent send reports disconnection rather than
  blocking forever.
- An external shutdown signal wakes blocked senders and receivers, which then report
  disconnection. A blocking operation must not be able to outlive a requested
  shutdown.
- The current number of queued items is observable, for the depth metric VE-015
  reports.
- Capacity is a small fixed constant chosen in the plan. It is not configurable.

### Integration state

- The binary remains runnable and behaviorally unchanged at the end of this pair.
  Execution stays serial; the queue may be unused by the binary or used trivially.
- No stage is moved onto a thread here.

## Constraints and non-goals

- No threads, no stage supervision, no shutdown orchestration. Those are VE-014.
- No drop policy, no overwrite policy, no capacity tuning, no configuration flags.
- No priority, no batching, no work stealing, no lock-free implementation.
- No instrumentation output beyond exposing the queue depth.

## Acceptance criteria

1. The four message types exist, carry the frame stamp, the decoded image, and the
   accumulated timings, and are moved rather than shared.
2. A compile-time test establishes that every message type can be moved between
   threads, or the stated owned-buffer fallback is implemented and its per-frame cost
   is documented.
3. The queue blocks on send when full and on receive when empty, demonstrated by
   tests that would hang or fail if it did not.
4. First-in, first-out order is preserved, and a concurrent producer and consumer
   exchange a large number of items with none lost, duplicated, or reordered.
5. Dropping all senders lets a receiver drain remaining items and then report
   disconnection; dropping the receiver makes a blocked sender report disconnection.
6. A shutdown signal wakes blocked senders and receivers, each reporting
   disconnection, with no test relying on a timeout to pass.
7. Queue depth is observable and matches the number of items sent but not received.
8. The binary builds and runs with unchanged behavior, and formatting, linting,
   tests, and the release build pass.
