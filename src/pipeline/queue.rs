use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

pub const QUEUE_CAPACITY: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Disconnected;

struct QueueInner<T> {
    items: VecDeque<T>,
    sender_count: usize,
    receiver_count: usize,
}

/// Type-erased wake target so shutdown can lock the same mutex waiters use.
trait QueueWaker: Send + Sync {
    fn wake_all(&self);
}

struct Shared<T> {
    inner: Mutex<QueueInner<T>>,
    not_empty: Condvar,
    not_full: Condvar,
    shutdown: Shutdown,
    capacity: usize,
}

impl<T: Send> QueueWaker for Shared<T> {
    fn wake_all(&self) {
        // Hold the wait mutex while notifying so a waiter cannot miss the signal
        // between checking a predicate and entering `wait`.
        let _guard = self.inner.lock().expect("queue mutex poisoned");
        self.not_empty.notify_all();
        self.not_full.notify_all();
    }
}

struct ShutdownInner {
    flag: AtomicBool,
    waiters: Mutex<Vec<Weak<dyn QueueWaker>>>,
}

pub struct Shutdown {
    inner: Arc<ShutdownInner>,
}

impl Shutdown {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ShutdownInner {
                flag: AtomicBool::new(false),
                waiters: Mutex::new(Vec::new()),
            }),
        }
    }

    pub fn request(&self) {
        self.inner.flag.store(true, Ordering::SeqCst);
        let waiters = self.inner.waiters.lock().expect("queue waiters poisoned");
        for waiter in waiters.iter().filter_map(Weak::upgrade) {
            waiter.wake_all();
        }
    }

    pub fn is_requested(&self) -> bool {
        self.inner.flag.load(Ordering::SeqCst)
    }

    pub fn clone_handle(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }

    fn register_waker(&self, waker: &Arc<dyn QueueWaker>) {
        let mut waiters = self.inner.waiters.lock().expect("queue waiters poisoned");
        waiters.retain(|existing| existing.strong_count() > 0);
        waiters.push(Arc::downgrade(waker));
    }
}

pub struct Sender<T> {
    shared: Arc<Shared<T>>,
}

pub struct Receiver<T> {
    shared: Arc<Shared<T>>,
}

#[derive(Clone)]
pub struct QueueDepthGauge<T> {
    shared: Arc<Shared<T>>,
}

impl<T> QueueDepthGauge<T> {
    pub fn snapshot(&self) -> (usize, usize) {
        let depth = self
            .shared
            .inner
            .lock()
            .expect("queue mutex poisoned")
            .items
            .len();
        (depth, self.shared.capacity)
    }
}

impl<T> Receiver<T> {
    pub fn depth_gauge(&self) -> QueueDepthGauge<T> {
        QueueDepthGauge {
            shared: Arc::clone(&self.shared),
        }
    }
}

pub fn bounded<T: Send + 'static>(
    capacity: usize,
    shutdown: &Shutdown,
) -> (Sender<T>, Receiver<T>) {
    let shared = Arc::new(Shared {
        inner: Mutex::new(QueueInner {
            items: VecDeque::with_capacity(capacity),
            sender_count: 1,
            receiver_count: 1,
        }),
        not_empty: Condvar::new(),
        not_full: Condvar::new(),
        shutdown: shutdown.clone_handle(),
        capacity,
    });

    let waker: Arc<dyn QueueWaker> = Arc::clone(&shared) as Arc<dyn QueueWaker>;
    shutdown.register_waker(&waker);

    (
        Sender {
            shared: Arc::clone(&shared),
        },
        Receiver { shared },
    )
}

impl<T> Sender<T> {
    pub fn send(&self, item: T) -> Result<(), Disconnected> {
        let shared = &self.shared;
        let mut inner = shared.inner.lock().expect("queue mutex poisoned");

        loop {
            if shared.shutdown.is_requested() {
                return Err(Disconnected);
            }

            if inner.receiver_count == 0 {
                return Err(Disconnected);
            }

            if inner.items.len() < shared.capacity {
                inner.items.push_back(item);
                shared.not_empty.notify_one();
                return Ok(());
            }

            inner = shared
                .not_full
                .wait(inner)
                .expect("not_full condvar poisoned");
        }
    }
}

impl<T> Receiver<T> {
    pub fn recv(&self) -> Result<T, Disconnected> {
        let shared = &self.shared;
        let mut inner = shared.inner.lock().expect("queue mutex poisoned");

        loop {
            if let Some(item) = inner.items.pop_front() {
                shared.not_full.notify_one();
                return Ok(item);
            }

            if shared.shutdown.is_requested() {
                return Err(Disconnected);
            }

            if inner.sender_count == 0 {
                return Err(Disconnected);
            }

            inner = shared
                .not_empty
                .wait(inner)
                .expect("not_empty condvar poisoned");
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn len(&self) -> usize {
        self.shared
            .inner
            .lock()
            .expect("queue mutex poisoned")
            .items
            .len()
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn capacity(&self) -> usize {
        self.shared.capacity
    }
}

impl<T> Drop for Sender<T> {
    fn drop(&mut self) {
        let shared = &self.shared;
        let mut inner = shared.inner.lock().expect("queue mutex poisoned");
        inner.sender_count = inner.sender_count.saturating_sub(1);
        shared.not_empty.notify_all();
        shared.not_full.notify_all();
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let shared = &self.shared;
        let mut inner = shared.inner.lock().expect("queue mutex poisoned");
        inner.receiver_count = inner.receiver_count.saturating_sub(1);
        shared.not_empty.notify_all();
        shared.not_full.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn recv_blocks_until_an_item_arrives_and_preserves_order() {
        let shutdown = Shutdown::new();
        let (tx, rx) = bounded::<u32>(2, &shutdown);
        let producer = thread::spawn(move || {
            for value in 0..3 {
                tx.send(value).unwrap();
            }
        });
        assert_eq!(rx.recv().unwrap(), 0);
        assert_eq!(rx.recv().unwrap(), 1);
        assert_eq!(rx.recv().unwrap(), 2);
        producer.join().unwrap();
    }

    #[test]
    fn send_blocks_while_the_queue_is_full() {
        let shutdown = Shutdown::new();
        let (tx, rx) = bounded::<u32>(2, &shutdown);
        let sent = Arc::new(AtomicUsize::new(0));
        let counter = Arc::clone(&sent);
        let producer = thread::spawn(move || {
            for value in 0..4 {
                tx.send(value).unwrap();
                counter.fetch_add(1, SeqCst);
            }
        });
        while sent.load(SeqCst) < 2 {
            std::hint::spin_loop();
        }
        assert!(sent.load(SeqCst) <= 3, "sender ran past the queue capacity");
        for expected in 0..4 {
            assert_eq!(rx.recv().unwrap(), expected);
        }
        producer.join().unwrap();
    }

    #[test]
    fn dropping_sender_lets_receiver_drain_then_disconnect() {
        let shutdown = Shutdown::new();
        let (tx, rx) = bounded::<u32>(2, &shutdown);
        tx.send(1).unwrap();
        tx.send(2).unwrap();
        drop(tx);
        assert_eq!(rx.recv().unwrap(), 1);
        assert_eq!(rx.recv().unwrap(), 2);
        assert_eq!(rx.recv(), Err(Disconnected));
    }

    #[test]
    fn dropping_receiver_disconnects_blocked_sender() {
        let shutdown = Shutdown::new();
        let (tx, rx) = bounded::<u32>(2, &shutdown);
        tx.send(0).unwrap();
        tx.send(1).unwrap();
        drop(rx);
        assert_eq!(tx.send(2), Err(Disconnected));
    }

    #[test]
    fn shutdown_wakes_a_blocked_receiver() {
        let shutdown = Shutdown::new();
        let (_tx, rx) = bounded::<u32>(2, &shutdown);
        let waiter = thread::spawn(move || rx.recv());
        shutdown.request();
        assert!(waiter.join().unwrap().is_err());
    }

    #[test]
    fn shutdown_wakes_a_blocked_sender() {
        let shutdown = Shutdown::new();
        let (tx, rx) = bounded::<u32>(2, &shutdown);
        tx.send(0).unwrap();
        tx.send(1).unwrap();
        let sender = thread::spawn(move || tx.send(2));
        shutdown.request();
        assert_eq!(sender.join().unwrap(), Err(Disconnected));
        drop(rx);
    }

    #[test]
    fn concurrent_producer_and_consumer_exchange_ten_thousand_items() {
        let shutdown = Shutdown::new();
        let (tx, rx) = bounded::<u32>(2, &shutdown);
        const COUNT: u32 = 10_000;

        let producer = thread::spawn(move || {
            for value in 0..COUNT {
                tx.send(value).unwrap();
            }
        });

        let mut received = 0_u32;
        while received < COUNT {
            let value = rx.recv().unwrap();
            assert_eq!(value, received);
            received += 1;
        }

        producer.join().unwrap();
        assert_eq!(received, COUNT);
    }

    #[test]
    fn len_and_capacity_reflect_queue_state() {
        let shutdown = Shutdown::new();
        let (tx, rx) = bounded::<u32>(4, &shutdown);
        assert_eq!(rx.capacity(), 4);
        assert_eq!(rx.len(), 0);

        tx.send(1).unwrap();
        tx.send(2).unwrap();
        tx.send(3).unwrap();
        assert_eq!(rx.len(), 3);

        assert_eq!(rx.recv().unwrap(), 1);
        assert_eq!(rx.len(), 2);
    }

    #[test]
    fn dropping_last_sender_unblocks_empty_receiver() {
        // Stress the check/wait race: receiver must not hang when the last sender
        // drops while the queue is empty.
        for _ in 0..200 {
            let shutdown = Shutdown::new();
            let (tx, rx) = bounded::<u32>(2, &shutdown);
            let receiver = thread::spawn(move || rx.recv());
            thread::sleep(Duration::from_micros(50));
            drop(tx);
            assert_eq!(
                receiver
                    .join()
                    .expect("receiver thread panicked")
                    .expect_err("receiver should disconnect"),
                Disconnected
            );
        }
    }
}
