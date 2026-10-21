//! Opt-in runtime profiling of a running tokenizer: how long each stage of the
//! encode pipeline takes, which path each input takes, how well each internal
//! cache works (the BPE word caches here, the segment and prefix caches per
//! tokenizer in `Tokenizer::observe_json`), the thread pools and the threads
//! doing the work, the slowest requests, and the Python API calls with their
//! GIL and lock waits.
//!
//! Compiled in only with the `observe` cargo feature. Without it every hook is
//! an empty inline function and [`Timer`], [`StageGuard`], [`JobGuard`] and
//! [`TaskGuard`] are zero-sized types without a destructor, so the default
//! build does no extra work.
//!
//! Counters are relaxed atomics read while writers keep going: a snapshot is not
//! one consistent instant, but every counter only grows, so the difference of
//! two snapshots divided by their `uptime_s` difference is a correct rate.
//! Histograms are log2 buckets: bucket `i` counts values in `[2^(i-1), 2^i)`
//! (microseconds for latencies, bytes for sizes), bucket 0 counts zeros.
//!
//! Per-request figures (stage times, the BPE cache counters) are summed in
//! thread-locals while a request runs and published when it ends, so the hot
//! loops pay for a local add, not an atomic. Hooks never panic: a thread whose
//! thread-locals are already being torn down is simply not counted.

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
    /// Answered whole by the prefix cache: an exact repeat, nothing encoded.
    PrefixExact,
}

#[cfg_attr(not(feature = "observe"), allow(dead_code))]
const PATHS: [&str; 6] = [
    "scan_whole",
    "scan_splits",
    "segments_cached",
    "fused_split",
    "general",
    "prefix_exact",
];

/// A timed stage of one encode. Stages do not overlap; what a request spends
/// outside all of them (allocation, glue) is its total minus their sum.
#[derive(Clone, Copy)]
pub enum Stage {
    /// Matching added (special) tokens and cutting the input at them.
    AddedTokens,
    /// The normalizer.
    Normalize,
    /// The prefix cache: finding a cached input sharing a prefix, storing one.
    PrefixCache,
    /// Pre-tokenization on its own (regex `Split`, `ByteLevel`, …).
    Pretokenize,
    /// The model (BPE); on scanner paths this includes the fused scan.
    Model,
    /// Looking text segments up in the segment cache (hashing included).
    CacheLookup,
    /// Inserting missed segments and assembling the cached and new ids.
    CacheAssemble,
    /// The post-processor (special tokens such as BOS/EOS).
    PostProcess,
}

#[cfg_attr(not(feature = "observe"), allow(dead_code))]
const STAGES: [&str; 8] = [
    "added_tokens",
    "normalize",
    "prefix_cache",
    "pretokenize",
    "model",
    "cache_lookup",
    "cache_assemble",
    "post_process",
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
    /// Reading `Encoding.ids`, which builds the Python list.
    EncodingIds,
    /// Reading `Encoding.attention_mask`, `type_ids` or `special_tokens_mask`.
    EncodingLists,
}

#[cfg_attr(not(feature = "observe"), allow(dead_code))]
const APIS: [&str; 9] = [
    "encode",
    "encode_ordinary",
    "encode_segments",
    "encode_batch",
    "encode_batch_flat",
    "decode",
    "decode_batch",
    "encoding_ids",
    "encoding_lists",
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

/// Counters of the BPE model's own caches, indexes into a per-thread array.
///
/// The fused pretoken cache (`fused_*`) serves the scanner and fused-`Split`
/// paths: every pretoken is looked up, a miss is merged (BPE) and inserted;
/// a pretoken longer than 15 bytes has no inline key and is always merged.
/// The general path (`general_*`) first tries the whole input as one token,
/// then this thread's cache, then the cache shared by all threads.
pub mod bpe {
    pub const FUSED_LOOKUPS: usize = 0;
    pub const FUSED_MISSES: usize = 1;
    pub const FUSED_MISS_BYTES: usize = 2;
    pub const FUSED_MERGE_NS: usize = 3;
    pub const FUSED_UNCACHEABLE: usize = 4;
    pub const FUSED_CLEARS: usize = 5;
    pub const GENERAL_CALLS: usize = 6;
    pub const GENERAL_EXACT: usize = 7;
    pub const GENERAL_THREAD_HITS: usize = 8;
    pub const GENERAL_SHARED_HITS: usize = 9;
    pub const GENERAL_MERGES: usize = 10;
    pub const GENERAL_MERGE_BYTES: usize = 11;
    pub const GENERAL_MERGE_NS: usize = 12;
    pub const N: usize = 13;
    #[cfg_attr(not(feature = "observe"), allow(dead_code))]
    pub(crate) const NAMES: [&str; N] = [
        "fused_lookups",
        "fused_misses",
        "fused_miss_bytes",
        "fused_merge_ns",
        "fused_uncacheable",
        "fused_clears",
        "general_calls",
        "general_exact",
        "general_thread_hits",
        "general_shared_hits",
        "general_merges",
        "general_merge_bytes",
        "general_merge_ns",
    ];
}

/// The bindings' tokenizer-state lock, taken for reading by every call and
/// for writing by every setter.
#[derive(Clone, Copy)]
pub enum Lock {
    Read,
    Write,
}

/// Number of log2 histogram buckets.
pub const BUCKETS: usize = 32;

/// How many of the slowest encodes are kept (see `reset_slow`).
pub const SLOWEST: usize = 32;

/// Log2 bucket of `v` (see the module docs).
#[inline]
pub fn bucket(v: u64) -> usize {
    ((u64::BITS - v.leading_zeros()) as usize).min(BUCKETS - 1)
}

#[cfg(feature = "observe")]
mod imp {
    use super::*;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering::Relaxed};
    use std::sync::{Mutex, OnceLock};
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
            if v != 0 {
                self.0.fetch_add(v, Relaxed);
            }
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

    /// One Python API entry point.
    struct ApiOp {
        calls: C,
        units: C,
        ns: C,
        rust_ns: C,
        released: C,
        gil_wait_ns: C,
        latency_us: Hist,
    }

    impl ApiOp {
        const fn new() -> Self {
            ApiOp {
                calls: C::new(),
                units: C::new(),
                ns: C::new(),
                rust_ns: C::new(),
                released: C::new(),
                gil_wait_ns: C::new(),
                latency_us: Hist::new(),
            }
        }
        fn json(&self) -> Value {
            json!({
                "calls": self.calls.get(),
                "units": self.units.get(),
                "ns": self.ns.get(),
                "rust_ns": self.rust_ns.get(),
                "gil_released": self.released.get(),
                "gil_wait_ns": self.gil_wait_ns.get(),
                "latency_us_log2": self.latency_us.json(),
            })
        }
    }

    /// One encode path: its encodes, their input and output, and where their
    /// time went.
    struct PathOp {
        calls: C,
        bytes: C,
        tokens: C,
        ns: C,
        stage_ns: [C; STAGES.len()],
        latency_us: Hist,
        size_bytes: Hist,
    }

    impl PathOp {
        const fn new() -> Self {
            PathOp {
                calls: C::new(),
                bytes: C::new(),
                tokens: C::new(),
                ns: C::new(),
                stage_ns: [const { C::new() }; STAGES.len()],
                latency_us: Hist::new(),
                size_bytes: Hist::new(),
            }
        }
        fn json(&self) -> Value {
            json!({
                "calls": self.calls.get(),
                "units": self.bytes.get(),
                "tokens": self.tokens.get(),
                "ns": self.ns.get(),
                "stage_ns": names(&STAGES, &self.stage_ns),
                "latency_us_log2": self.latency_us.json(),
                "size_bytes_log2": self.size_bytes.json(),
            })
        }
    }

    fn names(names: &[&str], counters: &[C]) -> Value {
        names
            .iter()
            .zip(counters)
            .map(|(n, c)| ((*n).to_owned(), c.get().into()))
            .collect::<serde_json::Map<_, _>>()
            .into()
    }

    static PATH_OPS: [PathOp; PATHS.len()] = [const { PathOp::new() }; PATHS.len()];
    static API_OPS: [ApiOp; APIS.len()] = [const { ApiOp::new() }; APIS.len()];
    static LOCK_WAITS: [(C, C); 2] = [const { (C::new(), C::new()) }; 2];

    /// A [`JobKind`]'s jobs, their tasks, the jobs' wall time and the tasks'
    /// summed time (their ratio is the average parallelism).
    struct JobOp {
        jobs: C,
        tasks: C,
        ns: C,
        task_ns: C,
    }

    static JOBS: [JobOp; JOB_KINDS.len()] = [const {
        JobOp {
            jobs: C::new(),
            tasks: C::new(),
            ns: C::new(),
            task_ns: C::new(),
        }
    }; JOB_KINDS.len()];

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
        encode_tokens: C,
        encode_ns: C,
        /// Encodes by [`Path`].
        paths: [C; PATHS.len()],
        /// Time of this thread's encodes by [`Stage`].
        stage_ns: [C; STAGES.len()],
        tasks: C,
        task_ns: C,
        api_calls: C,
        api_ns: C,
        /// The BPE caches' counters ([`bpe`]) of work done on this thread.
        bpe: [C; bpe::N],
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
                encode_tokens: C::new(),
                encode_ns: C::new(),
                paths: [const { C::new() }; PATHS.len()],
                stage_ns: [const { C::new() }; STAGES.len()],
                tasks: C::new(),
                task_ns: C::new(),
                api_calls: C::new(),
                api_ns: C::new(),
                bpe: [const { C::new() }; bpe::N],
            }
        }

        fn name(&self, i: usize) -> String {
            self.name
                .get()
                .cloned()
                .unwrap_or_else(|| format!("slot{i}"))
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
                "encode_tokens": self.encode_tokens.get(),
                "encode_ns": self.encode_ns.get(),
                "paths": names(&PATHS, &self.paths),
                "stage_ns": names(&STAGES, &self.stage_ns),
                "tasks": self.tasks.get(),
                "task_ns": self.task_ns.get(),
                "api_calls": self.api_calls.get(),
                "api_ns": self.api_ns.get(),
                "bpe": names(&bpe::NAMES, &self.bpe),
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
        /// The branch the running encode took (the last [`set_path`] wins).
        static PATH: Cell<u8> = const { Cell::new(Path::General as u8) };
        /// The running encode's time per [`Stage`].
        static REQ_STAGES: Cell<[u64; STAGES.len()]> = const { Cell::new([0; STAGES.len()]) };
        /// BPE cache counters not yet published to this thread's slot.
        static BPE: Cell<[u64; bpe::N]> = const { Cell::new([0; bpe::N]) };
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

    /// The calling thread's slot index, registering it on first use; `None`
    /// while the thread's locals are being destroyed.
    fn my_index() -> Option<usize> {
        ME.try_with(|m| {
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
        .ok()
    }

    fn me() -> Option<&'static Slot> {
        my_index().map(|i| &THREADS[i])
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

    // ── BPE cache counters ───────────────────────────────────────────────

    /// Add `v` to this thread's BPE counter `i` ([`bpe`]); published to the
    /// thread's slot when the current encode or pool task ends.
    #[inline]
    pub(crate) fn bpe_add(i: usize, v: u64) {
        let _ = BPE.try_with(|c| {
            let mut a = c.get();
            a[i] += v;
            c.set(a);
        });
    }

    /// Count a BPE cache miss of `bytes` whose merge took `ns`.
    #[inline]
    pub(crate) fn bpe_miss(misses: usize, bytes: usize, ns_counter: usize, ns: u64, len: usize) {
        let _ = BPE.try_with(|c| {
            let mut a = c.get();
            a[misses] += 1;
            a[bytes] += len as u64;
            a[ns_counter] += ns;
            c.set(a);
        });
    }

    fn publish_bpe(slot: &Slot) {
        let _ = BPE.try_with(|c| {
            let a = c.replace([0; bpe::N]);
            for (counter, v) in slot.bpe.iter().zip(a) {
                counter.add(v);
            }
        });
    }

    // ── stages and encodes ───────────────────────────────────────────────

    /// Adds its lifetime to the running encode's [`Stage`] when dropped.
    pub struct StageGuard(Stage, Instant);

    impl StageGuard {
        /// End the stage before the end of the scope.
        #[inline]
        pub fn end(self) {}
    }

    impl Drop for StageGuard {
        fn drop(&mut self) {
            let ns = self.1.elapsed().as_nanos() as u64;
            let _ = REQ_STAGES.try_with(|c| {
                let mut a = c.get();
                a[self.0 as usize] += ns;
                c.set(a);
            });
        }
    }

    /// Time a stage of the running encode until the guard drops.
    #[inline]
    pub(crate) fn stage(stage: Stage) -> StageGuard {
        StageGuard(stage, Instant::now())
    }

    /// Start counting an encode: forget stage times and the path left by
    /// pipeline calls made outside one.
    #[inline]
    pub(crate) fn begin_encode() {
        let _ = REQ_STAGES.try_with(|c| c.set([0; STAGES.len()]));
        let _ = PATH.try_with(|p| p.set(Path::General as u8));
    }

    /// Note the branch the running encode takes (the last call wins).
    #[inline]
    pub(crate) fn set_path(path: Path) {
        let _ = PATH.try_with(|p| p.set(path as u8));
    }

    const PATH_OF: [Path; PATHS.len()] = [
        Path::ScanWhole,
        Path::ScanSplits,
        Path::SegmentsCached,
        Path::FusedSplit,
        Path::General,
        Path::PrefixExact,
    ];

    /// Count the encode that just finished: `bytes` of input, `tokens` of
    /// output, `ns` in all, on the branch last given to [`set_path`], with
    /// the stage times gathered since [`begin_encode`].
    pub(crate) fn encode_done(bytes: usize, tokens: usize, ns: u64) {
        let path = PATH
            .try_with(|p| p.replace(Path::General as u8))
            .unwrap_or(Path::General as u8) as usize;
        let stages = REQ_STAGES
            .try_with(|c| c.replace([0; STAGES.len()]))
            .unwrap_or([0; STAGES.len()]);
        record_encode(PATH_OF[path], bytes, tokens, ns, &stages);
    }

    pub(crate) fn record_encode(
        path: Path,
        bytes: usize,
        tokens: usize,
        ns: u64,
        stages: &[u64; STAGES.len()],
    ) {
        let op = &PATH_OPS[path as usize];
        op.calls.add(1);
        op.bytes.add(bytes as u64);
        op.tokens.add(tokens as u64);
        op.ns.add(ns);
        op.latency_us.add(ns / 1000);
        op.size_bytes.add(bytes as u64);
        for (c, v) in op.stage_ns.iter().zip(stages) {
            c.add(*v);
        }
        let slot = my_index();
        if let Some(i) = slot {
            let s = &THREADS[i];
            s.encodes.add(1);
            s.encode_bytes.add(bytes as u64);
            s.encode_tokens.add(tokens as u64);
            s.encode_ns.add(ns);
            s.paths[path as usize].add(1);
            for (c, v) in s.stage_ns.iter().zip(stages) {
                c.add(*v);
            }
            publish_bpe(s);
        }
        slow_candidate(ns, || Slow {
            at_s: epoch().elapsed().as_secs_f64(),
            ns,
            bytes: bytes as u64,
            tokens: tokens as u64,
            path: path as u8,
            stages: *stages,
            slot: slot.unwrap_or(SLOTS - 1),
        });
    }

    // ── the slowest encodes ──────────────────────────────────────────────

    struct Slow {
        at_s: f64,
        ns: u64,
        bytes: u64,
        tokens: u64,
        path: u8,
        stages: [u64; STAGES.len()],
        slot: usize,
    }

    static SLOW: Mutex<Vec<Slow>> = Mutex::new(Vec::new());
    /// The fastest kept encode's time once [`SLOWEST`] are kept, else 0: a
    /// faster encode is turned away without taking the lock.
    static SLOW_FLOOR: AtomicU64 = AtomicU64::new(0);

    fn slow_candidate(ns: u64, make: impl FnOnce() -> Slow) {
        if ns <= SLOW_FLOOR.load(Relaxed) {
            return;
        }
        let Ok(mut kept) = SLOW.lock() else {
            return;
        };
        kept.push(make());
        kept.sort_unstable_by_key(|s| std::cmp::Reverse(s.ns));
        kept.truncate(SLOWEST);
        let floor = if kept.len() == SLOWEST {
            kept[SLOWEST - 1].ns
        } else {
            0
        };
        SLOW_FLOOR.store(floor, Relaxed);
    }

    /// Forget the slowest encodes kept so far (to start a profile window).
    pub fn reset_slow() {
        if let Ok(mut kept) = SLOW.lock() {
            kept.clear();
        }
        SLOW_FLOOR.store(0, Relaxed);
    }

    fn slow_json() -> Value {
        let kept = match SLOW.lock() {
            Ok(k) => k,
            Err(e) => e.into_inner(),
        };
        kept.iter()
            .map(|s| {
                let stages: serde_json::Map<_, _> = STAGES
                    .iter()
                    .zip(s.stages)
                    .map(|(n, v)| ((*n).to_owned(), v.into()))
                    .collect();
                json!({
                    "at_s": s.at_s,
                    "ns": s.ns,
                    "bytes": s.bytes,
                    "tokens": s.tokens,
                    "path": PATHS[s.path as usize],
                    "stage_ns": stages,
                    "thread": THREADS[s.slot].name(s.slot),
                })
            })
            .collect::<Vec<_>>()
            .into()
    }

    // ── pools ────────────────────────────────────────────────────────────

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
            let op = &JOBS[self.kind as usize];
            op.jobs.add(1);
            op.tasks.add(self.tasks as u64);
            op.ns.add(self.start.elapsed().as_nanos() as u64);
        }
    }

    /// Counts one task of a [`JobKind`] job on the running thread when dropped,
    /// and publishes the BPE counters of its work. A thread of rayon's global
    /// pool (which `encode_batch` uses) is labelled a worker the first time it
    /// runs one.
    pub struct TaskGuard(JobKind, Instant);

    impl TaskGuard {
        #[inline]
        pub fn start(kind: JobKind) -> Self {
            TaskGuard(kind, Instant::now())
        }
    }

    impl Drop for TaskGuard {
        fn drop(&mut self) {
            let ns = self.1.elapsed().as_nanos() as u64;
            JOBS[self.0 as usize].task_ns.add(ns);
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
                publish_bpe(s);
            }
        }
    }

    // ── Python API ───────────────────────────────────────────────────────

    /// A Python API call of `units` (input bytes, or ids for decoding and
    /// list building) that took `ns`, `rust_ns` of it in Rust (state lock
    /// included), with the GIL released during the Rust work or not, and
    /// `gil_wait_ns` spent getting the GIL back afterwards.
    pub fn api_call(
        api: Api,
        units: usize,
        ns: u64,
        rust_ns: u64,
        gil_released: bool,
        gil_wait_ns: u64,
    ) {
        let op = &API_OPS[api as usize];
        op.calls.add(1);
        op.units.add(units as u64);
        op.ns.add(ns);
        op.rust_ns.add(rust_ns);
        op.gil_wait_ns.add(gil_wait_ns);
        op.latency_us.add(ns / 1000);
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

    // ── snapshots ────────────────────────────────────────────────────────

    /// The calling thread's own counters (as in the snapshot's `threads`),
    /// after publishing its pending BPE counters.
    pub fn current_thread() -> Value {
        let Some(i) = my_index() else {
            return Value::Null;
        };
        let s = &THREADS[i];
        publish_bpe(s);
        s.json(i)
    }

    pub fn snapshot() -> Value {
        let n = NEXT.load(Relaxed).min(SLOTS);
        let threads: Vec<Value> = THREADS[..n]
            .iter()
            .enumerate()
            .map(|(i, s)| s.json(i))
            .collect();
        let mut bpe_total = [0u64; bpe::N];
        for s in &THREADS[..n] {
            for (t, c) in bpe_total.iter_mut().zip(&s.bpe) {
                *t += c.get();
            }
        }
        let jobs: serde_json::Map<_, _> = JOB_KINDS
            .iter()
            .zip(&JOBS)
            .map(|(n, o)| {
                (
                    (*n).to_owned(),
                    json!({"jobs": o.jobs.get(), "tasks": o.tasks.get(), "ns": o.ns.get(), "task_ns": o.task_ns.get()}),
                )
            })
            .collect();
        let paths: serde_json::Map<_, _> = PATHS
            .iter()
            .zip(&PATH_OPS)
            .map(|(n, o)| ((*n).to_owned(), o.json()))
            .collect();
        let apis: serde_json::Map<_, _> = APIS
            .iter()
            .zip(&API_OPS)
            .map(|(n, o)| ((*n).to_owned(), o.json()))
            .collect();
        let bpe: serde_json::Map<_, _> = bpe::NAMES
            .iter()
            .zip(bpe_total)
            .map(|(n, v)| ((*n).to_owned(), v.into()))
            .collect();
        json!({
            "enabled": true,
            "uptime_s": epoch().elapsed().as_secs_f64(),
            "pool": {
                "bpe_threads": crate::pre_tokenized::bpe_pool_threads(),
                "global_threads": rayon::current_num_threads(),
            },
            "stages": STAGES,
            "jobs": jobs,
            "paths": paths,
            "api": apis,
            "bpe": bpe,
            "lock_waits": {
                "read": {"count": LOCK_WAITS[0].0.get(), "ns": LOCK_WAITS[0].1.get()},
                "write": {"count": LOCK_WAITS[1].0.get(), "ns": LOCK_WAITS[1].1.get()},
            },
            "slowest": slow_json(),
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

    pub struct StageGuard;
    pub struct JobGuard;
    pub struct TaskGuard;

    impl StageGuard {
        #[inline(always)]
        pub fn end(self) {}
    }

    impl JobGuard {
        #[inline(always)]
        pub fn start(_: JobKind, _: usize) -> Self {
            JobGuard
        }
    }

    impl TaskGuard {
        #[inline(always)]
        pub fn start(_: JobKind) -> Self {
            TaskGuard
        }
    }

    #[inline(always)]
    pub fn current_thread_named() -> bool {
        true
    }
    #[inline(always)]
    pub fn name_current_thread(_: &str) {}
    #[inline(always)]
    pub(crate) fn bpe_add(_: usize, _: u64) {}
    #[inline(always)]
    pub(crate) fn bpe_miss(_: usize, _: usize, _: usize, _: u64, _: usize) {}
    #[inline(always)]
    pub(crate) fn stage(_: Stage) -> StageGuard {
        StageGuard
    }
    #[inline(always)]
    pub(crate) fn begin_encode() {}
    #[inline(always)]
    pub(crate) fn set_path(_: Path) {}
    #[inline(always)]
    pub(crate) fn encode_done(_: usize, _: usize, _: u64) {}
    #[inline(always)]
    pub fn api_call(_: Api, _: usize, _: u64, _: u64, _: bool, _: u64) {}
    #[inline(always)]
    pub fn lock_wait(_: Lock, _: u64) {}
    #[inline(always)]
    pub fn reset_slow() {}
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
        assert_eq!(std::mem::size_of::<StageGuard>(), 0);
        assert!(!std::mem::needs_drop::<JobGuard>());
        assert!(!std::mem::needs_drop::<TaskGuard>());
        assert!(!std::mem::needs_drop::<StageGuard>());
        assert_eq!(std::mem::size_of::<Timer>(), 0);
        assert_eq!(Timer::start().ns(), 0);
    }

    #[cfg(feature = "observe")]
    mod enabled {
        use super::super::*;
        use std::sync::{Arc, Barrier};

        fn n(v: &Value, path: &[&str]) -> u64 {
            path.iter()
                .fold(v, |v, k| &v[*k])
                .as_u64()
                .unwrap_or_else(|| panic!("{path:?} missing"))
        }

        #[test]
        fn snapshot_has_every_section() {
            let s = snapshot();
            assert_eq!(s["enabled"], true);
            assert!(s["uptime_s"].as_f64().unwrap() >= 0.0);
            assert!(n(&s, &["pool", "bpe_threads"]) >= 1);
            assert!(n(&s, &["pool", "global_threads"]) >= 1);
            assert_eq!(s["stages"].as_array().unwrap().len(), STAGES.len());
            for k in JOB_KINDS {
                assert!(s["jobs"][k]["task_ns"].is_u64(), "{k}");
            }
            for k in PATHS {
                assert_eq!(
                    s["paths"][k]["latency_us_log2"].as_array().unwrap().len(),
                    BUCKETS
                );
                assert_eq!(
                    s["paths"][k]["stage_ns"].as_object().unwrap().len(),
                    STAGES.len()
                );
            }
            for k in APIS {
                assert!(s["api"][k]["rust_ns"].is_u64(), "{k}");
            }
            for k in bpe::NAMES {
                assert!(s["bpe"][k].is_u64(), "{k}");
            }
            assert!(s["slowest"].is_array());
            assert!(s["lock_waits"]["read"]["count"].is_u64());
            let text = s.to_string();
            assert_eq!(
                serde_json::from_str::<Value>(&text).unwrap()["enabled"],
                true
            );
        }

        #[test]
        fn per_thread_counters_are_exact() {
            std::thread::Builder::new()
                .name("observe-exact".into())
                .spawn(|| {
                    for i in 0..10u64 {
                        record_encode(Path::General, 100 + i as usize, 3, 1_000, &[1; 8]);
                    }
                    api_call(Api::Decode, 5, 2_000, 1_500, false, 0);
                    drop(TaskGuard::start(JobKind::Batch));
                    bpe_add(bpe::FUSED_LOOKUPS, 7);
                    bpe_miss(
                        bpe::FUSED_MISSES,
                        bpe::FUSED_MISS_BYTES,
                        bpe::FUSED_MERGE_NS,
                        50,
                        4,
                    );
                    let me = current_thread();
                    assert_eq!(me["name"], "observe-exact");
                    assert_eq!(me["kind"], "caller");
                    assert_eq!(me["alive"], true);
                    assert_eq!(n(&me, &["encodes"]), 10);
                    assert_eq!(n(&me, &["encode_bytes"]), (100..110).sum::<u64>());
                    assert_eq!(n(&me, &["encode_tokens"]), 30);
                    assert_eq!(n(&me, &["encode_ns"]), 10_000);
                    assert_eq!(n(&me, &["paths", "general"]), 10);
                    assert_eq!(n(&me, &["paths", "scan_whole"]), 0);
                    assert_eq!(n(&me, &["api_calls"]), 1);
                    assert_eq!(n(&me, &["api_ns"]), 2_000);
                    assert_eq!(n(&me, &["tasks"]), 1);
                    assert_eq!(n(&me, &["bpe", "fused_lookups"]), 7);
                    assert_eq!(n(&me, &["bpe", "fused_misses"]), 1);
                    assert_eq!(n(&me, &["bpe", "fused_miss_bytes"]), 4);
                    assert_eq!(n(&me, &["bpe", "fused_merge_ns"]), 50);
                    assert!(me["os_tid"].as_u64().unwrap() > 0);
                })
                .unwrap()
                .join()
                .unwrap();
        }

        #[test]
        fn stages_and_path_belong_to_the_encode_that_ran_them() {
            // On a thread of its own, read through that thread's counters:
            // other tests encode on other threads meanwhile.
            std::thread::spawn(|| {
                let before = current_thread();
                // Left over by pipeline calls outside an encode: forgotten.
                set_path(Path::ScanWhole);
                stage(Stage::Model).end();
                begin_encode();
                set_path(Path::FusedSplit);
                {
                    let _s = stage(Stage::Pretokenize);
                    std::thread::sleep(std::time::Duration::from_millis(20));
                }
                {
                    let _s = stage(Stage::Model);
                    std::thread::sleep(std::time::Duration::from_millis(2));
                }
                encode_done(10, 2, 50_000_000);
                let after = current_thread();
                let d = |k: &[&str]| n(&after, k) - n(&before, k);
                assert_eq!(d(&["paths", "fused_split"]), 1);
                assert_eq!(d(&["encode_tokens"]), 2);
                assert!(d(&["stage_ns", "pretokenize"]) >= 20_000_000);
                // Each stage holds its own time only; the sleeps are far
                // enough apart that a slow wake-up cannot blur them.
                let model = d(&["stage_ns", "model"]);
                assert!((2_000_000..20_000_000).contains(&model), "{model}");
                // The next encode starts from zero and the General default.
                begin_encode();
                encode_done(1, 1, 1);
                let me = current_thread();
                assert_eq!(n(&me, &["paths", "general"]), 1);
                assert_eq!(n(&me, &["paths", "fused_split"]), 1);
                assert_eq!(n(&me, &["paths", "scan_whole"]), 0);
            })
            .join()
            .unwrap();
        }

        #[test]
        fn exited_threads_are_marked_dead_and_keep_their_counts() {
            let h = std::thread::Builder::new()
                .name("observe-exits".into())
                .spawn(|| {
                    record_encode(Path::ScanWhole, 42, 1, 1, &[0; 8]);
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
        fn jobs_count_tasks_and_parallel_time() {
            let before = snapshot();
            {
                let _job = JobGuard::start(JobKind::RegexSplit, 4);
                std::thread::scope(|s| {
                    for _ in 0..4 {
                        s.spawn(|| {
                            let _t = TaskGuard::start(JobKind::RegexSplit);
                            std::thread::sleep(std::time::Duration::from_millis(3));
                        });
                    }
                });
            }
            let after = snapshot();
            let d = |k: &str| {
                n(&after, &["jobs", "regex_split", k]) - n(&before, &["jobs", "regex_split", k])
            };
            assert!(d("jobs") >= 1);
            assert!(d("tasks") >= 4);
            assert!(d("ns") >= 3_000_000);
            // Four tasks of 3 ms ran side by side: more task time than wall time.
            assert!(d("task_ns") >= 12_000_000);
        }

        #[test]
        fn api_latency_lands_in_its_bucket() {
            let before = snapshot();
            api_call(Api::EncodeBatchFlat, 10, 3_000_000, 2_000_000, true, 500);
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
            assert!(d("rust_ns") >= 2_000_000);
            assert!(d("units") >= 10);
        }

        #[test]
        fn the_slowest_encodes_are_kept_in_order() {
            // Big enough to outrank whatever other tests record meanwhile.
            let base = 1u64 << 50;
            std::thread::Builder::new()
                .name("observe-slow".into())
                .spawn(move || {
                    for i in 0..(SLOWEST as u64 + 10) {
                        let mut st = [0u64; 8];
                        st[Stage::Model as usize] = i;
                        record_encode(Path::ScanSplits, i as usize, 1, base + i, &st);
                    }
                })
                .unwrap()
                .join()
                .unwrap();
            let s = snapshot();
            let kept = s["slowest"].as_array().unwrap();
            assert_eq!(kept.len(), SLOWEST);
            let ns: Vec<u64> = kept.iter().map(|k| n(k, &["ns"])).collect();
            assert!(ns.windows(2).all(|w| w[0] >= w[1]), "{ns:?}");
            assert_eq!(ns[0], base + SLOWEST as u64 + 9);
            assert_eq!(kept[0]["path"], "scan_splits");
            assert_eq!(kept[0]["thread"], "observe-slow");
            assert_eq!(n(&kept[0], &["stage_ns", "model"]), SLOWEST as u64 + 9);
            reset_slow();
            // Other tests' encodes may refill the list at once, but none of
            // these is kept.
            assert!(
                snapshot()["slowest"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .all(|k| n(k, &["ns"]) < base)
            );
        }

        #[test]
        fn counters_only_grow_under_concurrency() {
            let threads = 8;
            let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
            let start = Arc::new(Barrier::new(threads + 1));
            let handles: Vec<_> = (0..threads)
                .map(|i| {
                    let (stop, start) = (stop.clone(), start.clone());
                    std::thread::spawn(move || {
                        start.wait();
                        while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                            begin_encode();
                            set_path(Path::ScanSplits);
                            stage(Stage::Model).end();
                            bpe_add(bpe::FUSED_LOOKUPS, 3);
                            encode_done(i + 1, 1, 10);
                            api_call(Api::Encode, i + 1, 10, 5, true, 1);
                            let _job = JobGuard::start(JobKind::Batch, 2);
                            drop(TaskGuard::start(JobKind::Batch));
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
                        // Slowest-request lists are replaced, not grown.
                        if matches!(k.as_str(), "alive" | "kind" | "name" | "slowest") {
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
