//! Opt-in runtime counters for looking inside a running tokenizer: the thread
//! pools, the threads doing the work, which encode path each input takes, the
//! Python API calls and their waits, and (per tokenizer) the segment cache
//! shard by shard (see `Tokenizer::observe_json`).
//!
//! Compiled in only with the `observe` cargo feature. Without it every hook is
//! an empty inline function and [`Timer`], [`JobGuard`] and [`TaskGuard`] are
//! zero-sized types without a destructor, so the default build does no extra
//! work.
//!
//! Counters are relaxed atomics read while writers keep going: a snapshot is not
//! one consistent instant, but every counter only grows, so the difference of
//! two snapshots divided by their `uptime_s` difference is a correct rate.
//! Histograms are log2 buckets: bucket `i` counts values in `[2^(i-1), 2^i)`
//! (microseconds for latencies, bytes for sizes), bucket 0 counts zeros.
//!
//! Hooks never panic: a thread whose thread-locals are already being torn down
//! is simply not counted.

use serde_json::{Value, json};

/// Which branch of the encode pipeline an input took.
#[derive(Clone, Copy)]
pub enum Path {
    /// Scanner fast path, the whole input one plain-text split.
    ScanWhole,
    /// Scanner fast path, text splits between added tokens (chat templates).
    ScanSplits,
    /// Text splits through the segment cache.
    SegmentsCached,
    /// `Split` pre-tokenizer fused with the model, no scanner.
    FusedSplit,
    /// Normalize + pre-tokenize + model pipeline.
    General,
}

#[cfg_attr(not(feature = "observe"), allow(dead_code))]
const PATHS: [&str; 5] = [
    "scan_whole",
    "scan_splits",
    "segments_cached",
    "fused_split",
    "general",
];

/// A Python API entry point (the bindings report these).
#[derive(Clone, Copy)]
pub enum Api {
    Encode,
    EncodeOrdinary,
    EncodeSegments,
    EncodeBatch,
    EncodeBatchFlat,
    Decode,
    DecodeBatch,
}

#[cfg_attr(not(feature = "observe"), allow(dead_code))]
const APIS: [&str; 7] = [
    "encode",
    "encode_ordinary",
    "encode_segments",
    "encode_batch",
    "encode_batch_flat",
    "decode",
    "decode_batch",
];

/// Why a parallel job was started.
#[derive(Clone, Copy)]
pub enum JobKind {
    /// Fused scan + BPE of one long plain-text split, cut at newlines.
    Scan,
    /// Fused scan + BPE of chat-template text splits, packed into jobs.
    ScanSplits,
    /// Pre-tokenized splits tokenized in parallel (non-scanner models).
    Pretokenized,
    /// A long input's regex `Split` matched in parallel chunks.
    RegexSplit,
    /// One input per task (`encode_batch`, `encode_batch_flat`).
    Batch,
}

#[cfg_attr(not(feature = "observe"), allow(dead_code))]
const JOB_KINDS: [&str; 5] = [
    "scan",
    "chat_splits",
    "pretokenized",
    "regex_split",
    "batch",
];

/// The bindings' tokenizer-state lock, taken for reading by every call and
/// for writing by every setter.
#[derive(Clone, Copy)]
pub enum Lock {
    Read,
    Write,
}

/// Number of log2 histogram buckets.
pub const BUCKETS: usize = 32;

/// Log2 bucket of `v` (see the module docs).
#[inline]
pub fn bucket(v: u64) -> usize {
    ((u64::BITS - v.leading_zeros()) as usize).min(BUCKETS - 1)
}

#[cfg(feature = "observe")]
mod imp {
    use super::*;
    use std::cell::Cell;
    use std::sync::OnceLock;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering::Relaxed};
    use std::time::Instant;

    pub const ENABLED: bool = true;

    /// Wall-clock stopwatch for a hook.
    pub struct Timer(Instant);

    impl Timer {
        #[inline]
        pub fn start() -> Self {
            Timer(Instant::now())
        }

        #[inline]
        pub fn ns(&self) -> u64 {
            self.0.elapsed().as_nanos() as u64
        }
    }

    fn epoch() -> Instant {
        static EPOCH: OnceLock<Instant> = OnceLock::new();
        *EPOCH.get_or_init(Instant::now)
    }

    struct C(AtomicU64);

    impl C {
        const fn new() -> Self {
            C(AtomicU64::new(0))
        }
        #[inline]
        fn add(&self, v: u64) {
            self.0.fetch_add(v, Relaxed);
        }
        fn get(&self) -> u64 {
            self.0.load(Relaxed)
        }
    }

    struct Hist([C; BUCKETS]);

    impl Hist {
        const fn new() -> Self {
            Hist([const { C::new() }; BUCKETS])
        }
        #[inline]
        fn add(&self, v: u64) {
            self.0[bucket(v)].add(1);
        }
        fn json(&self) -> Value {
            Value::from(self.0.iter().map(C::get).collect::<Vec<_>>())
        }
    }

    /// Calls, bytes (or ids) and time of one encode path or API.
    struct Op {
        calls: C,
        units: C,
        ns: C,
        released: C,
        gil_wait_ns: C,
        latency_us: Hist,
    }

    impl Op {
        const fn new() -> Self {
            Op {
                calls: C::new(),
                units: C::new(),
                ns: C::new(),
                released: C::new(),
                gil_wait_ns: C::new(),
                latency_us: Hist::new(),
            }
        }
        fn record(&self, units: u64, ns: u64) {
            self.calls.add(1);
            self.units.add(units);
            self.ns.add(ns);
            self.latency_us.add(ns / 1000);
        }
        fn json(&self) -> Value {
            json!({
                "calls": self.calls.get(),
                "units": self.units.get(),
                "ns": self.ns.get(),
                "gil_released": self.released.get(),
                "gil_wait_ns": self.gil_wait_ns.get(),
                "latency_us_log2": self.latency_us.json(),
            })
        }
    }

    static PATH_OPS: [Op; PATHS.len()] = [const { Op::new() }; PATHS.len()];
    static API_OPS: [Op; APIS.len()] = [const { Op::new() }; APIS.len()];
    static LOCK_WAITS: [(C, C); 2] = [const { (C::new(), C::new()) }; 2];
    /// Jobs, tasks and job wall time by [`JobKind`].
    static JOBS: [(C, C, C); JOB_KINDS.len()] =
        [const { (C::new(), C::new(), C::new()) }; JOB_KINDS.len()];

    /// Per-thread counters; each slot is written by its own thread only.
    #[repr(align(128))]
    struct Slot {
        /// 0 free, 1 caller, 2 pool worker.
        kind: AtomicU8,
        alive: AtomicBool,
        name: OnceLock<String>,
        os_tid: AtomicU64,
        encodes: C,
        encode_bytes: C,
        encode_ns: C,
        /// Encodes by [`Path`].
        paths: [C; PATHS.len()],
        tasks: C,
        task_ns: C,
        api_calls: C,
        api_ns: C,
    }

    impl Slot {
        const fn new() -> Self {
            Slot {
                kind: AtomicU8::new(0),
                alive: AtomicBool::new(false),
                name: OnceLock::new(),
                os_tid: AtomicU64::new(0),
                encodes: C::new(),
                encode_bytes: C::new(),
                encode_ns: C::new(),
                paths: [const { C::new() }; PATHS.len()],
                tasks: C::new(),
                task_ns: C::new(),
                api_calls: C::new(),
                api_ns: C::new(),
            }
        }

        fn json(&self, i: usize) -> Value {
            json!({
                "slot": i,
                "kind": match self.kind.load(Relaxed) { 2 => "worker", _ => "caller" },
                "alive": self.alive.load(Relaxed),
                "name": self.name.get().cloned().unwrap_or_default(),
                "os_tid": self.os_tid.load(Relaxed),
                "encodes": self.encodes.get(),
                "encode_bytes": self.encode_bytes.get(),
                "encode_ns": self.encode_ns.get(),
                "paths": PATHS.iter().zip(&self.paths).map(|(n, c)| ((*n).to_owned(), c.get().into())).collect::<serde_json::Map<_, _>>(),
                "tasks": self.tasks.get(),
                "task_ns": self.task_ns.get(),
                "api_calls": self.api_calls.get(),
                "api_ns": self.api_ns.get(),
            })
        }
    }

    /// Threads beyond this many share the last slot ("(other threads)").
    pub(crate) const SLOTS: usize = 256;
    static THREADS: [Slot; SLOTS] = [const { Slot::new() }; SLOTS];
    static NEXT: AtomicUsize = AtomicUsize::new(0);

    /// Marks the thread's slot dead when the thread exits.
    struct Exit(usize);

    impl Drop for Exit {
        fn drop(&mut self) {
            if self.0 < SLOTS - 1 {
                THREADS[self.0].alive.store(false, Relaxed);
            }
        }
    }

    thread_local! {
        static ME: Cell<usize> = const { Cell::new(usize::MAX) };
        static EXIT: Cell<Option<Exit>> = const { Cell::new(None) };
    }

    /// This thread's id as the OS shows it (`top -H`, `py-spy`, Instruments).
    fn os_tid() -> u64 {
        #[cfg(target_os = "macos")]
        {
            let mut tid = 0u64;
            // SAFETY: a null thread means the calling thread; `tid` is a valid
            // out pointer.
            unsafe { libc::pthread_threadid_np(0, &mut tid) };
            tid
        }
        #[cfg(target_os = "linux")]
        {
            // "/proc/thread-self" links to "/proc/<pid>/task/<tid>".
            std::fs::read_link("/proc/thread-self")
                .ok()
                .and_then(|p| p.file_name()?.to_str()?.parse().ok())
                .unwrap_or(0)
        }
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        0
    }

    /// The calling thread's slot, registering it on first use; `None` while
    /// the thread's locals are being destroyed.
    fn me() -> Option<&'static Slot> {
        let i = ME
            .try_with(|m| {
                if m.get() == usize::MAX {
                    let i = NEXT.fetch_add(1, Relaxed).min(SLOTS - 1);
                    let slot = &THREADS[i];
                    if i == SLOTS - 1 {
                        let _ = slot.name.set("(other threads)".to_owned());
                    } else {
                        slot.kind.store(1, Relaxed);
                        slot.os_tid.store(os_tid(), Relaxed);
                        if let Some(name) = std::thread::current().name() {
                            let _ = slot.name.set(name.to_owned());
                        }
                        let _ = EXIT.try_with(|e| e.set(Some(Exit(i))));
                    }
                    slot.alive.store(true, Relaxed);
                    m.set(i);
                }
                m.get()
            })
            .ok()?;
        Some(&THREADS[i])
    }

    /// Whether the calling thread already has a name in the snapshot.
    pub fn current_thread_named() -> bool {
        me().is_none_or(|s| s.name.get().is_some())
    }

    /// Name the calling thread in the snapshot (the first name sticks).
    pub fn name_current_thread(name: &str) {
        if let Some(s) = me() {
            let _ = s.name.set(name.to_owned());
        }
    }

    /// Mark the calling thread as a pool worker (the BPE pool's start handler).
    pub(crate) fn register_worker() {
        if let Some(s) = me() {
            s.kind.store(2, Relaxed);
        }
    }

    /// Counts a parallel job of `tasks` tasks under its [`JobKind`], with its
    /// wall time, when dropped.
    pub struct JobGuard {
        kind: JobKind,
        tasks: usize,
        start: Instant,
    }

    impl JobGuard {
        #[inline]
        pub fn start(kind: JobKind, tasks: usize) -> Self {
            JobGuard {
                kind,
                tasks,
                start: Instant::now(),
            }
        }
    }

    impl Drop for JobGuard {
        fn drop(&mut self) {
            let (jobs, n, ns) = &JOBS[self.kind as usize];
            jobs.add(1);
            n.add(self.tasks as u64);
            ns.add(self.start.elapsed().as_nanos() as u64);
        }
    }

    /// Counts one task of a parallel job on the running thread when dropped.
    /// A thread of rayon's global pool (which `encode_batch` uses) is labelled
    /// a worker the first time it runs one.
    pub struct TaskGuard(Instant);

    impl TaskGuard {
        #[inline]
        pub fn start() -> Self {
            TaskGuard(Instant::now())
        }
    }

    impl Drop for TaskGuard {
        fn drop(&mut self) {
            let ns = self.0.elapsed().as_nanos() as u64;
            if let Some(s) = me() {
                if s.kind.load(Relaxed) == 1
                    && s.name.get().is_none()
                    && let Some(i) = rayon::current_thread_index()
                {
                    s.kind.store(2, Relaxed);
                    let _ = s.name.set(format!("rayon-{i}"));
                }
                s.tasks.add(1);
                s.task_ns.add(ns);
            }
        }
    }

    thread_local! {
        /// The branch the current encode took, set by the pipeline on the way.
        static PATH: Cell<u8> = const { Cell::new(Path::General as u8) };
    }

    /// Note the branch the running encode takes (the last call wins).
    #[inline]
    pub(crate) fn set_path(path: Path) {
        let _ = PATH.try_with(|p| p.set(path as u8));
    }

    /// Count the encode that just finished: `bytes` of input in `ns`, on the
    /// branch last given to [`set_path`] (`General` if none was).
    pub(crate) fn encode_done(bytes: usize, ns: u64) {
        let path = PATH
            .try_with(|p| p.replace(Path::General as u8))
            .unwrap_or(Path::General as u8);
        let path = [
            Path::ScanWhole,
            Path::ScanSplits,
            Path::SegmentsCached,
            Path::FusedSplit,
            Path::General,
        ][path as usize];
        encode_path(path, bytes, ns);
    }

    pub(crate) fn encode_path(path: Path, bytes: usize, ns: u64) {
        PATH_OPS[path as usize].record(bytes as u64, ns);
        if let Some(s) = me() {
            s.encodes.add(1);
            s.encode_bytes.add(bytes as u64);
            s.encode_ns.add(ns);
            s.paths[path as usize].add(1);
        }
    }

    /// A Python API call of `units` (input bytes, or ids for decoding) that
    /// took `ns`, with the GIL released during the Rust work or not, and
    /// `gil_wait_ns` spent getting the GIL back afterwards.
    pub fn api_call(api: Api, units: usize, ns: u64, gil_released: bool, gil_wait_ns: u64) {
        let op = &API_OPS[api as usize];
        op.record(units as u64, ns);
        op.gil_wait_ns.add(gil_wait_ns);
        if gil_released {
            op.released.add(1);
        }
        if let Some(s) = me() {
            s.api_calls.add(1);
            s.api_ns.add(ns);
        }
    }

    /// A wait of `ns` for the bindings' state lock (only waits that blocked).
    pub fn lock_wait(lock: Lock, ns: u64) {
        let (n, t) = &LOCK_WAITS[lock as usize];
        n.add(1);
        t.add(ns);
    }

    /// The calling thread's own counters (as in the snapshot's `threads`).
    pub fn current_thread() -> Value {
        let Some(s) = me() else {
            return Value::Null;
        };
        let i = ME.try_with(Cell::get).unwrap_or(SLOTS - 1).min(SLOTS - 1);
        s.json(i)
    }

    pub fn snapshot() -> Value {
        let ops = |names: &[&str], ops: &[Op]| -> Value {
            names
                .iter()
                .zip(ops)
                .map(|(n, o)| ((*n).to_owned(), o.json()))
                .collect::<serde_json::Map<_, _>>()
                .into()
        };
        let threads: Vec<Value> = THREADS[..NEXT.load(Relaxed).min(SLOTS)]
            .iter()
            .enumerate()
            .map(|(i, s)| s.json(i))
            .collect();
        let jobs: serde_json::Map<_, _> = JOB_KINDS
            .iter()
            .zip(&JOBS)
            .map(|(n, (j, t, ns))| {
                (
                    (*n).to_owned(),
                    json!({"jobs": j.get(), "tasks": t.get(), "ns": ns.get()}),
                )
            })
            .collect();
        json!({
            "enabled": true,
            "uptime_s": epoch().elapsed().as_secs_f64(),
            "pool": {
                "bpe_threads": crate::pre_tokenized::bpe_pool_threads(),
                "global_threads": rayon::current_num_threads(),
            },
            "jobs": jobs,
            "paths": ops(&PATHS, &PATH_OPS),
            "api": ops(&APIS, &API_OPS),
            "lock_waits": {
                "read": {"count": LOCK_WAITS[0].0.get(), "ns": LOCK_WAITS[0].1.get()},
                "write": {"count": LOCK_WAITS[1].0.get(), "ns": LOCK_WAITS[1].1.get()},
            },
            "threads": threads,
        })
    }
}

#[cfg(not(feature = "observe"))]
mod imp {
    use super::*;

    pub const ENABLED: bool = false;

    pub struct Timer;

    impl Timer {
        #[inline(always)]
        pub fn start() -> Self {
            Timer
        }
        #[inline(always)]
        pub fn ns(&self) -> u64 {
            0
        }
    }

    #[inline(always)]
    pub fn current_thread_named() -> bool {
        true
    }
    #[inline(always)]
    pub fn name_current_thread(_: &str) {}
    pub struct JobGuard;

    impl JobGuard {
        #[inline(always)]
        pub fn start(_: JobKind, _: usize) -> Self {
            JobGuard
        }
    }

    pub struct TaskGuard;

    impl TaskGuard {
        #[inline(always)]
        pub fn start() -> Self {
            TaskGuard
        }
    }
    #[inline(always)]
    pub(crate) fn set_path(_: Path) {}
    #[inline(always)]
    pub(crate) fn encode_done(_: usize, _: u64) {}
    #[inline(always)]
    pub fn api_call(_: Api, _: usize, _: u64, _: bool, _: u64) {}
    #[inline(always)]
    pub fn lock_wait(_: Lock, _: u64) {}
    pub fn current_thread() -> Value {
        Value::Null
    }
    pub fn snapshot() -> Value {
        json!({ "enabled": false })
    }
}

pub use imp::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buckets_are_log2() {
        assert_eq!(bucket(0), 0);
        assert_eq!(bucket(1), 1);
        assert_eq!(bucket(2), 2);
        assert_eq!(bucket(3), 2);
        assert_eq!(bucket(4), 3);
        assert_eq!(bucket(1023), 10);
        assert_eq!(bucket(1024), 11);
        assert_eq!(bucket(u64::MAX), BUCKETS - 1);
    }

    #[cfg(not(feature = "observe"))]
    #[test]
    fn disabled_build_reports_nothing() {
        const { assert!(!ENABLED) };
        assert_eq!(snapshot(), json!({"enabled": false}));
        assert_eq!(current_thread(), Value::Null);
        assert_eq!(std::mem::size_of::<JobGuard>(), 0);
        assert_eq!(std::mem::size_of::<TaskGuard>(), 0);
        assert!(!std::mem::needs_drop::<JobGuard>());
        assert!(!std::mem::needs_drop::<TaskGuard>());
        assert_eq!(std::mem::size_of::<Timer>(), 0);
        assert_eq!(Timer::start().ns(), 0);
    }

    #[cfg(feature = "observe")]
    mod enabled {
        use super::super::*;
        use std::sync::{Arc, Barrier};

        fn n(v: &Value, path: &[&str]) -> u64 {
            path.iter().fold(v, |v, k| &v[*k]).as_u64().unwrap()
        }

        #[test]
        fn snapshot_has_every_section() {
            let s = snapshot();
            assert_eq!(s["enabled"], true);
            assert!(s["uptime_s"].as_f64().unwrap() >= 0.0);
            assert!(n(&s, &["pool", "bpe_threads"]) >= 1);
            assert!(n(&s, &["pool", "global_threads"]) >= 1);
            for k in JOB_KINDS {
                assert!(s["jobs"][k]["jobs"].is_u64(), "{k}");
            }
            for k in PATHS {
                assert_eq!(
                    s["paths"][k]["latency_us_log2"].as_array().unwrap().len(),
                    BUCKETS
                );
            }
            for k in APIS {
                assert!(s["api"][k]["calls"].is_u64(), "{k}");
            }
            assert!(s["lock_waits"]["read"]["count"].is_u64());
            // Round-trips as JSON text.
            let text = s.to_string();
            assert_eq!(
                serde_json::from_str::<Value>(&text).unwrap()["enabled"],
                true
            );
        }

        #[test]
        fn per_thread_counters_are_exact() {
            // A fresh thread has a slot of its own, so exact counts hold even
            // while other tests run.
            std::thread::Builder::new()
                .name("observe-exact".into())
                .spawn(|| {
                    for i in 0..10 {
                        encode_path(Path::General, 100 + i, 1_000);
                    }
                    api_call(Api::Decode, 5, 2_000, false, 0);
                    drop(TaskGuard::start());
                    let me = current_thread();
                    assert_eq!(me["name"], "observe-exact");
                    assert_eq!(me["kind"], "caller");
                    assert_eq!(me["alive"], true);
                    assert_eq!(n(&me, &["encodes"]), 10);
                    assert_eq!(n(&me, &["encode_bytes"]), (100..110).sum::<u64>());
                    assert_eq!(n(&me, &["encode_ns"]), 10_000);
                    assert_eq!(n(&me, &["paths", "general"]), 10);
                    assert_eq!(n(&me, &["paths", "scan_whole"]), 0);
                    assert_eq!(n(&me, &["api_calls"]), 1);
                    assert_eq!(n(&me, &["api_ns"]), 2_000);
                    assert_eq!(n(&me, &["tasks"]), 1);
                    assert!(me["os_tid"].as_u64().unwrap() > 0);
                })
                .unwrap()
                .join()
                .unwrap();
        }

        #[test]
        fn exited_threads_are_marked_dead_and_keep_their_counts() {
            let h = std::thread::Builder::new()
                .name("observe-exits".into())
                .spawn(|| {
                    encode_path(Path::ScanWhole, 42, 1);
                    current_thread()["slot"].as_u64().unwrap()
                })
                .unwrap();
            let slot = h.join().unwrap() as usize;
            let s = snapshot();
            let t = &s["threads"][slot];
            assert_eq!(t["name"], "observe-exits");
            assert_eq!(t["alive"], false);
            assert_eq!(n(t, &["encodes"]), 1);
            assert_eq!(n(t, &["encode_bytes"]), 42);
        }

        #[test]
        fn naming_keeps_the_first_name() {
            std::thread::spawn(|| {
                assert!(!current_thread_named());
                name_current_thread("first");
                name_current_thread("second");
                assert!(current_thread_named());
                assert_eq!(current_thread()["name"], "first");
            })
            .join()
            .unwrap();
        }

        #[test]
        fn jobs_count_tasks_and_time() {
            let before = snapshot();
            {
                let _job = JobGuard::start(JobKind::RegexSplit, 4);
                std::thread::sleep(std::time::Duration::from_millis(2));
            }
            let after = snapshot();
            let d = |k: &str| {
                n(&after, &["jobs", "regex_split", k]) - n(&before, &["jobs", "regex_split", k])
            };
            assert!(d("jobs") >= 1);
            assert!(d("tasks") >= 4);
            assert!(d("ns") >= 2_000_000);
        }

        #[test]
        fn api_latency_lands_in_its_bucket() {
            let before = snapshot();
            api_call(Api::EncodeBatchFlat, 10, 3_000_000, true, 500);
            let after = snapshot();
            let h = |s: &Value| {
                s["api"]["encode_batch_flat"]["latency_us_log2"][bucket(3_000)]
                    .as_u64()
                    .unwrap()
            };
            assert!(h(&after) > h(&before));
            let d = |k: &str| {
                n(&after, &["api", "encode_batch_flat", k])
                    - n(&before, &["api", "encode_batch_flat", k])
            };
            assert!(d("gil_released") >= 1);
            assert!(d("gil_wait_ns") >= 500);
            assert!(d("units") >= 10);
        }

        #[test]
        fn counters_only_grow_under_concurrency() {
            // Writers on 8 threads while snapshots are taken: every counter of
            // a later snapshot is at least the earlier one's.
            let threads = 8;
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let start = Arc::new(Barrier::new(threads + 1));
            let handles: Vec<_> = (0..threads)
                .map(|i| {
                    let (stop, start) = (stop.clone(), start.clone());
                    std::thread::spawn(move || {
                        start.wait();
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            encode_path(Path::ScanSplits, i + 1, 10);
                            api_call(Api::Encode, i + 1, 10, true, 1);
                            let _job = JobGuard::start(JobKind::Batch, 2);
                            drop(TaskGuard::start());
                            lock_wait(Lock::Read, 1);
                        }
                    })
                })
                .collect();
            start.wait();
            let mut prev = snapshot();
            for _ in 0..200 {
                let cur = snapshot();
                assert_monotonic(&prev, &cur, "");
                prev = cur;
            }
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
            for h in handles {
                h.join().unwrap();
            }
        }

        fn assert_monotonic(a: &Value, b: &Value, at: &str) {
            match (a, b) {
                (Value::Object(x), Value::Object(y)) => {
                    for (k, v) in x {
                        if k == "alive" || k == "kind" || k == "name" {
                            continue;
                        }
                        assert_monotonic(v, &y[k], &format!("{at}.{k}"));
                    }
                }
                (Value::Array(x), Value::Array(y)) => {
                    assert!(y.len() >= x.len(), "{at} shrank");
                    for (i, (v, w)) in x.iter().zip(y).enumerate() {
                        assert_monotonic(v, w, &format!("{at}[{i}]"));
                    }
                }
                (Value::Number(x), Value::Number(y)) => {
                    if let (Some(x), Some(y)) = (x.as_u64(), y.as_u64()) {
                        assert!(y >= x, "{at} went from {x} to {y}");
                    } else {
                        assert!(y.as_f64().unwrap() >= x.as_f64().unwrap(), "{at}");
                    }
                }
                _ => {}
            }
        }
    }
}
