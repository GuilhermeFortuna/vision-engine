use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};

pub const QUEUE_CAPACITY: usize = 2;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Disconnected;

struct QueueWake {
    not_empty: Condvar,
    not_full: Condvar,
}

struct ShutdownInner {
    flag: AtomicBool,
    queue_wakes: Mutex<Vec<Weak<QueueWake>>>,
}

pub struct Shutdown {
    inner: Arc<ShutdownInner>,
}

impl Shutdown {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(ShutdownInner {
                flag: AtomicBool::new(false),
                queue_wakes: Mutex::new(Vec::new()),
            }),
        }
    }

    pub fn request(&self) {
        self.inner.flag.store(true, Ordering::SeqCst);
        let wakes = self.inner.queue_wakes.lock().expect("queue wakes poisoned");
        for wake in wakes.iter().filter_map(Weak::upgrade) {
            wake.not_empty.notify_all();
            wake.not_full.notify_all();
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

    fn register_queue_wake(&self, wake: &Arc<QueueWake>) {
        let mut wakes = self.inner.queue_wakes.lock().expect("queue wakes poisoned");
        wakes.retain(|existing| existing.strong_count() > 0);
        wakes.push(Arc::downgrade(wake));
    }
}

struct Shared<T> {
    queue: Mutex<VecDeque<T>>,
    wake: Arc<QueueWake>,
    sender_count: Mutex<usize>,
    receiver_count: Mutex<usize>,
    shutdown: Shutdown,
    capacity: usize,
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
            .queue
            .lock()
            .expect("queue mutex poisoned")
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

pub fn bounded<T>(capacity: usize, shutdown: &Shutdown) -> (Sender<T>, Receiver<T>) {
    let wake = Arc::new(QueueWake {
        not_empty: Condvar::new(),
        not_full: Condvar::new(),
    });
    shutdown.register_queue_wake(&wake);

    let shared = Arc::new(Shared {
        queue: Mutex::new(VecDeque::with_capacity(capacity)),
        wake,
        sender_count: Mutex::new(1),
        receiver_count: Mutex::new(1),
        shutdown: shutdown.clone_handle(),
        capacity,
    });

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
        let mut queue = shared.queue.lock().expect("queue mutex poisoned");

        loop {
            if shared.shutdown.is_requested() {
                return Err(Disconnected);
            }

            if *shared
                .receiver_count
                .lock()
                .expect("receiver count poisoned")
                == 0
            {
                return Err(Disconnected);
            }

            if queue.len() < shared.capacity {
                queue.push_back(item);
                shared.wake.not_empty.notify_one();
                return Ok(());
            }

            queue = shared
                .wake
                .not_full
                .wait(queue)
                .expect("not_full condvar poisoned");
        }
    }
}

impl<T> Receiver<T> {
    pub fn recv(&self) -> Result<T, Disconnected> {
        let shared = &self.shared;
        let mut queue = shared.queue.lock().expect("queue mutex poisoned");

        loop {
            if let Some(item) = queue.pop_front() {
                shared.wake.not_full.notify_one();
                return Ok(item);
            }

            if shared.shutdown.is_requested() {
                return Err(Disconnected);
            }

            if *shared.sender_count.lock().expect("sender count poisoned") == 0 {
                return Err(Disconnected);
            }

            queue = shared
                .wake
                .not_empty
                .wait(queue)
                .expect("not_empty condvar poisoned");
        }
    }

    #[cfg_attr(not(test), allow(dead_code))]
    pub fn len(&self) -> usize {
        self.shared
            .queue
            .lock()
            .expect("queue mutex poisoned")
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
        let mut sender_count = shared.sender_count.lock().expect("sender count poisoned");
        *sender_count = sender_count.saturating_sub(1);
        drop(sender_count);

        shared.wake.not_empty.notify_all();
        shared.wake.not_full.notify_all();
    }
}

impl<T> Drop for Receiver<T> {
    fn drop(&mut self) {
        let shared = &self.shared;
        let mut receiver_count = shared
            .receiver_count
            .lock()
            .expect("receiver count poisoned");
        *receiver_count = receiver_count.saturating_sub(1);
        drop(receiver_count);

        shared.wake.not_empty.notify_all();
        shared.wake.not_full.notify_all();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering::SeqCst};
    use std::thread;

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
}
