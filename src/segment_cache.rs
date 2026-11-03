//! Opt-in cache of the token ids of text segments between added tokens.
//!
//! Chat-template output, the input a serving stack encodes, is text segments
//! separated by added tokens (`<|im_start|>`, `<|im_end|>`, tool-call
//! markers, …). A chat or coding agent re-sends its whole history on every
//! request, so all but the newest segments were already encoded by an earlier
//! request, but not as a byte prefix of it: templates drop the reasoning of
//! earlier turns, and agents compact their history into a summary, so the
//! prefix cache cannot reuse them. This cache reuses them by content.
//!
//! A segment's ids depend on nothing but its normalized text: added tokens are
//! hard boundaries, the normalizer runs per segment, and every pre-tokenizer in
//! this crate and the model act on each split on its own, independent of its
//! position in the input (a pre-tokenizer that treated the first split
//! differently, like HF's `Metaspace` with `prepend_scheme: first`, would break
//! this; none is implemented). Every hit is confirmed against the stored text,
//! so a hash collision is a miss, never a wrong answer, and the result is
//! identical to encoding from scratch. Through `encode`, an input that is a
//! single segment is not looked up, so plain-text traffic is unaffected;
//! `encode_segments` looks up each untrusted segment on its own, unless it is
//! too long to ever be cached.
//!
//! Segment text is often written by whoever sends the request, so the maps are
//! keyed by a randomly keyed SipHash of the content hash rather than by the
//! content hash itself, which is unkeyed and easy to invert: crafted text
//! cannot choose its shard or bucket, and so cannot pile entries into one
//! probe chain.
//!
//! The cache is sharded (concurrent sessions hit it from the thread pool, so
//! one lock would serialize them), each shard behind a read-write lock: a hit
//! on a young entry and a miss change nothing, so concurrent lookups share the
//! lock, and only moving an old entry back and inserting take it exclusively.
//! Each shard and its lock fill their own cache lines, so lookups in different
//! shards do not write a line another shard's lock or counters are on. There
//! is a byte budget per shard, and the cache evicts
//! with two generations rather than an LRU, which would need a linked list
//! over thousands of entries: entries are inserted into a young map; when the
//! next insert would overfill it, the old map is dropped and the young one
//! becomes old. A hit on an old entry moves it back into the young map, so
//! recently used segments survive. Both generations stay within the shard's
//! share of the budget, so the cache as a whole stays within `max_bytes`. A segment costing
//! more than 1/64 of the budget is never cached (with a 1 MiB budget, one of
//! more than about 8 KB of text), so the budget should be sized for the
//! longest segments, tool outputs typically, not just for the total.

use std::{
    collections::HashMap,
    hash::{BuildHasher, BuildHasherDefault, Hasher, RandomState},
    sync::{
        Arc, RwLock, RwLockReadGuard, RwLockWriteGuard,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

/// Number of independently locked shards.
const SHARDS: usize = 16;

/// Bookkeeping bytes charged per entry on top of its text and ids. A counting
/// allocator measures 70 bytes requested per entry whatever its size: the map
/// slot, in a table between half and seven-eighths full, and the `Arc` header
/// of the ids. What the allocator rounds that to is its own: glibc stays
/// within 94 bytes at every size, while macOS reaches 103 for an entry of a
/// few hundred bytes of text and rounds a large entry much further (20 KB of
/// text and ids as 20.5 KB, 100 KB as 128 KB). 96 covers glibc at every size
/// and macOS for small entries; it does not cover macOS's rounding of large
/// ones, because the budget accounts for the data, so
/// `SegmentCacheStats::bytes` under-reports their memory there.
const ENTRY_OVERHEAD: usize = 96;

/// 64-bit hash of `bytes`.
///
/// Four multiply-rotate lanes over 32-byte blocks keep the loop from being a
/// single dependency chain, and a murmur3 finalizer mixes every output bit.
/// It is not collision-resistant and need not be: a hit is confirmed by
/// comparing the stored text, so a collision only costs a cache miss. Nor is
/// it keyed, so it never indexes a map directly; see [`SegmentCache::key`].
pub(crate) fn hash_bytes(bytes: &[u8]) -> u64 {
    const K: u64 = 0x9E37_79B9_7F4A_7C15;
    #[inline(always)]
    fn round(acc: u64, word: u64) -> u64 {
        (acc ^ word).wrapping_mul(K).rotate_left(29)
    }
    #[inline(always)]
    fn word(bytes: &[u8]) -> u64 {
        let mut buf = [0u8; 8];
        buf[..bytes.len()].copy_from_slice(bytes);
        u64::from_le_bytes(buf)
    }

    let mut lanes: [u64; 4] = [
        0x243F_6A88_85A3_08D3,
        0x1319_8A2E_0370_7344,
        0xA409_3822_299F_31D0,
        0x082E_FA98_EC4E_6C89,
    ];
    let mut i = 0;
    while i + 32 <= bytes.len() {
        for (k, lane) in lanes.iter_mut().enumerate() {
            *lane = round(*lane, word(&bytes[i + 8 * k..i + 8 * k + 8]));
        }
        i += 32;
    }
    let mut h = bytes.len() as u64;
    for lane in lanes {
        h = round(h, lane);
    }
    for chunk in bytes[i..].chunks(8) {
        h = round(h, word(chunk));
    }
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 33;
    h = h.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    h ^ (h >> 33)
}

/// Map hasher for keys that are already [`hash_bytes`] outputs.
#[derive(Default)]
pub(crate) struct PassThrough(u64);

impl Hasher for PassThrough {
    fn finish(&self) -> u64 {
        self.0
    }
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 << 8) | u64::from(b);
        }
    }
    fn write_u64(&mut self, n: u64) {
        self.0 = n;
    }
}

pub(crate) type HashMapU64<V> = HashMap<u64, V, BuildHasherDefault<PassThrough>>;

struct Entry {
    text: Box<[u8]>,
    ids: Arc<[u32]>,
}

impl Entry {
    fn cost(&self) -> usize {
        cost_of(self.text.len(), self.ids.len())
    }
}

/// Bytes charged for an entry of `text_len` bytes of text and `n_ids` ids.
fn cost_of(text_len: usize, n_ids: usize) -> usize {
    text_len + n_ids * 4 + ENTRY_OVERHEAD
}

/// One lock's worth of the cache: two generations of entries, and the hit and
/// miss counts of the lookups that came to it, so threads working on
/// different shards do not contend on one counter. The counts are atomic
/// because most lookups hold the lock only for reading. Aligned so that a
/// shard and its lock start on their own cache line.
#[derive(Default)]
#[repr(align(64))]
struct Shard {
    young: HashMapU64<Entry>,
    old: HashMapU64<Entry>,
    young_bytes: usize,
    old_bytes: usize,
    hits: AtomicU64,
    misses: AtomicU64,
    hit_bytes: AtomicU64,
    miss_bytes: AtomicU64,
    #[cfg(feature = "observe")]
    obs: ShardObs,
}

/// A shard's [`crate::observe`] counters: atomic where a lookup under the
/// read lock updates them, plain where only the write lock does.
#[cfg(feature = "observe")]
#[derive(Default)]
struct ShardObs {
    /// Lookups whose hash matched an entry of different text.
    collisions: AtomicU64,
    /// Old-generation hits moved back into the young generation.
    promotions: u64,
    inserts: u64,
    /// Inserts that replaced an entry with the same hash.
    replaced: u64,
    /// Young generations turned old, each dropping the previous old one.
    rotations: u64,
    evicted_entries: u64,
    evicted_bytes: u64,
    /// Lock acquisitions (read or write) that had to wait, and how long they
    /// waited.
    contended: AtomicU64,
    wait_ns: AtomicU64,
    /// Segment sizes of every [`SIZE_SAMPLE`]th lookup of a thread: counting
    /// every lookup inside the lock lengthens the critical section enough to
    /// show when many threads share the cache.
    hit_sizes: [AtomicU64; crate::observe::BUCKETS],
    miss_sizes: [AtomicU64; crate::observe::BUCKETS],
}

#[cfg(feature = "observe")]
impl ShardObs {
    /// Count a lock acquisition that had to wait since `start`.
    fn waited(&self, start: std::time::Instant) {
        self.contended.fetch_add(1, Relaxed);
        self.wait_ns
            .fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
    }
}

/// One lookup in this many per thread lands in the size histograms.
#[cfg(feature = "observe")]
const SIZE_SAMPLE: u32 = 16;

#[cfg(feature = "observe")]
thread_local! {
    static LOOKUPS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Cache-wide [`crate::observe`] counters (segments turned away before any
/// shard is locked, and misses are tokenized outside the cache).
#[cfg(feature = "observe")]
#[derive(Default)]
struct CacheObs {
    rejected: AtomicU64,
    rejected_bytes: AtomicU64,
    miss_tokenize_ns: AtomicU64,
}

impl Shard {
    /// Insert into the young generation, first turning it old if `entry`
    /// would not fit. Keeps `young_bytes <= budget` for entries costing at
    /// most `budget`. Returns the old generation this dropped, if any, for the
    /// caller to free after releasing the shard's lock.
    #[must_use]
    fn insert_young(
        &mut self,
        hash: u64,
        entry: Entry,
        budget: usize,
    ) -> Option<HashMapU64<Entry>> {
        if let Some(replaced) = self.young.remove(&hash) {
            self.young_bytes -= replaced.cost();
            #[cfg(feature = "observe")]
            {
                self.obs.replaced += 1;
            }
        }
        let mut dropped = None;
        if self.young_bytes + entry.cost() > budget {
            #[cfg(feature = "observe")]
            {
                self.obs.rotations += 1;
                self.obs.evicted_entries += self.old.len() as u64;
                self.obs.evicted_bytes += self.old_bytes as u64;
            }
            dropped = Some(std::mem::replace(
                &mut self.old,
                std::mem::take(&mut self.young),
            ));
            self.old_bytes = std::mem::take(&mut self.young_bytes);
        }
        self.young_bytes += entry.cost();
        self.young.insert(hash, entry);
        dropped
    }

    fn remove_old(&mut self, hash: u64) -> Option<Entry> {
        let entry = self.old.remove(&hash)?;
        self.old_bytes -= entry.cost();
        Some(entry)
    }

    /// Count a lookup of `len` bytes of text as a hit or a miss.
    fn count(&self, hit: bool, len: usize) {
        let (count, bytes) = match hit {
            true => (&self.hits, &self.hit_bytes),
            false => (&self.misses, &self.miss_bytes),
        };
        count.fetch_add(1, Relaxed);
        bytes.fetch_add(len as u64, Relaxed);
    }

    /// The lookup of `text`, whose key is `hash`, as far as it can go without
    /// changing anything, counted: `Some` with the cached ids or `None` (a
    /// miss) once it is settled, or `None` without counting when `text` is an
    /// old entry, which only [`Self::get`] can move back.
    fn peek(&self, hash: u64, text: &[u8]) -> Option<Option<Arc<[u32]>>> {
        let ids = match self.young.get(&hash) {
            Some(entry) => {
                let ids = (*entry.text == *text).then(|| entry.ids.clone());
                #[cfg(feature = "observe")]
                if ids.is_none() {
                    self.obs.collisions.fetch_add(1, Relaxed);
                }
                ids
            }
            None => match self.old.get(&hash) {
                Some(entry) if *entry.text == *text => return None,
                _other => {
                    #[cfg(feature = "observe")]
                    if _other.is_some() {
                        self.obs.collisions.fetch_add(1, Relaxed);
                    }
                    None
                }
            },
        };
        self.count(ids.is_some(), text.len());
        Some(ids)
    }

    /// The cached ids of `text`, whose key is `hash`, counted as a hit or a
    /// miss. A hit on an old entry moves it back into the young generation,
    /// which may drop the old one: that is returned for the caller to free
    /// after releasing the shard's lock.
    fn get(
        &mut self,
        hash: u64,
        text: &[u8],
        budget: usize,
    ) -> (Option<Arc<[u32]>>, Option<HashMapU64<Entry>>) {
        let mut dropped = None;
        let ids = match self.young.get(&hash) {
            Some(entry) => {
                let ids = (*entry.text == *text).then(|| entry.ids.clone());
                #[cfg(feature = "observe")]
                if ids.is_none() {
                    self.obs.collisions.fetch_add(1, Relaxed);
                }
                ids
            }
            None if self
                .old
                .get(&hash)
                .is_some_and(|entry| *entry.text == *text) =>
            {
                let entry = self.remove_old(hash).unwrap();
                let ids = entry.ids.clone();
                dropped = self.insert_young(hash, entry, budget);
                #[cfg(feature = "observe")]
                {
                    self.obs.promotions += 1;
                }
                Some(ids)
            }
            None => {
                #[cfg(feature = "observe")]
                if self.old.contains_key(&hash) {
                    self.obs.collisions.fetch_add(1, Relaxed);
                }
                None
            }
        };
        self.count(ids.is_some(), text.len());
        (ids, dropped)
    }

    /// Put a lookup of `len` bytes in the size histograms ([`SIZE_SAMPLE`]).
    #[cfg(feature = "observe")]
    fn sample(&self, hit: bool, len: usize) {
        let b = crate::observe::bucket(len as u64);
        let sizes = match hit {
            true => &self.obs.hit_sizes,
            false => &self.obs.miss_sizes,
        };
        sizes[b].fetch_add(1, Relaxed);
    }
}

/// A snapshot of a segment cache's activity and size, from
/// [`crate::Tokenizer::segment_cache_stats`].
///
/// Every text segment of an input that added tokens split is a hit or a miss
/// (an input without any is neither), as is every untrusted
/// [`crate::EncodeSegment`] short enough to be cached: a hit's ids were
/// reused, a miss was tokenized.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SegmentCacheStats {
    /// Segments whose ids were reused.
    pub hits: u64,
    /// Segments that were tokenized.
    pub misses: u64,
    /// Bytes of segment text the hits covered.
    pub hit_bytes: u64,
    /// Bytes of segment text the misses covered.
    pub miss_bytes: u64,
    /// Segments currently cached.
    pub entries: usize,
    /// Bytes currently charged against the budget: text, ids and a fixed
    /// overhead per entry.
    pub bytes: usize,
    /// The budget the cache was created with.
    pub max_bytes: usize,
}

/// Content-addressed cache of text-segment encodings. See the module docs.
/// A tokenizer holds it in an `Arc` so that its copies can share it
/// ([`crate::Tokenizer::share_segment_cache_from`]).
pub(crate) struct SegmentCache {
    shards: Box<[RwLock<Shard>]>,
    /// Byte budget of each shard's young generation; the old generation is at
    /// most as large, so the shards together stay within `max_bytes`.
    young_budget: usize,
    max_bytes: usize,
    /// Random SipHash keys for [`Self::key`].
    keys: RandomState,
    #[cfg(feature = "observe")]
    obs: CacheObs,
}

impl SegmentCache {
    /// A cache holding at most `max_bytes` of segment text and ids.
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| RwLock::default()).collect(),
            young_budget: (max_bytes / SHARDS / 2).max(1),
            max_bytes,
            keys: RandomState::new(),
            #[cfg(feature = "observe")]
            obs: CacheObs::default(),
        }
    }

    /// The budget the cache was created with.
    pub(crate) fn max_bytes(&self) -> usize {
        self.max_bytes
    }

    /// Append the ids of the segment `text` to `out`: the cached ones, or
    /// those `tokenize` appends, which are then cached.
    pub(crate) fn encode<E>(
        &self,
        text: &str,
        out: &mut Vec<u32>,
        tokenize: impl FnOnce(&mut Vec<u32>) -> Result<(), E>,
    ) -> Result<(), E> {
        use crate::observe::{Stage, stage};
        let lookup = stage(Stage::CacheLookup);
        let hash = self.key(text.as_bytes());
        let found = self.get(hash, text.as_bytes());
        lookup.end();
        if let Some(ids) = found {
            let _stage = stage(Stage::CacheAssemble);
            out.extend_from_slice(&ids);
            return Ok(());
        }
        let start = out.len();
        // The miss's own stages (the model's) are timed inside `tokenize`.
        let timer = crate::observe::Timer::start();
        tokenize(out)?;
        self.record_miss_tokenize(timer.ns());
        let _stage = stage(Stage::CacheAssemble);
        self.insert(hash, text.as_bytes(), &out[start..]);
        Ok(())
    }

    /// Whether a segment of `len` bytes of text can be cached at all: one whose
    /// text alone is over the size limit (see [`Self::insert`]) never is.
    pub(crate) fn can_cache(&self, len: usize) -> bool {
        cost_of(len, 0) <= self.young_budget / 2
    }

    /// The map key of `text`: its [`hash_bytes`] through SipHash with this
    /// cache's random keys. Rehashing 8 bytes costs a few nanoseconds; keying
    /// by `hash_bytes` directly would let whoever writes the text choose its
    /// shard and bucket.
    fn key(&self, text: &[u8]) -> u64 {
        self.keys.hash_one(hash_bytes(text))
    }

    fn shard(&self, hash: u64) -> &RwLock<Shard> {
        // Bits the maps' bucket index (low) and tag (top) do not use.
        &self.shards[(hash >> 32) as usize % SHARDS]
    }

    /// Read-lock the shard of `hash`, counting a wait if a writer held it.
    fn read_shard(&self, hash: u64) -> RwLockReadGuard<'_, Shard> {
        let lock = self.shard(hash);
        #[cfg(feature = "observe")]
        match lock.try_read() {
            Ok(shard) => return shard,
            Err(std::sync::TryLockError::WouldBlock) => {
                let t = std::time::Instant::now();
                let shard = lock.read().unwrap();
                shard.obs.waited(t);
                return shard;
            }
            // Poisoned: the plain lock below panics as it always did.
            Err(std::sync::TryLockError::Poisoned(_)) => {}
        }
        lock.read().unwrap()
    }

    /// Write-lock the shard of `hash`, counting a wait if another thread held
    /// it.
    fn write_shard(&self, hash: u64) -> RwLockWriteGuard<'_, Shard> {
        let lock = self.shard(hash);
        #[cfg(feature = "observe")]
        match lock.try_write() {
            Ok(shard) => return shard,
            Err(std::sync::TryLockError::WouldBlock) => {
                let t = std::time::Instant::now();
                let shard = lock.write().unwrap();
                shard.obs.waited(t);
                return shard;
            }
            // Poisoned: the plain lock below panics as it always did.
            Err(std::sync::TryLockError::Poisoned(_)) => {}
        }
        lock.write().unwrap()
    }

    /// The cached ids of `text`, whose [`Self::key`] is `hash`, counted as a
    /// hit or a miss.
    fn get(&self, hash: u64, text: &[u8]) -> Option<Arc<[u32]>> {
        #[cfg(feature = "observe")]
        let sample = LOOKUPS
            .try_with(|n| {
                let k = n.get().wrapping_add(1);
                n.set(k);
                k % SIZE_SAMPLE == 0
            })
            .unwrap_or(false);
        // A young hit and a miss are settled under the read lock; only an old
        // hit, which moves the entry back to the young generation, takes the
        // write lock, and looks again: another thread may have moved, replaced
        // or evicted it in between.
        {
            let shard = self.read_shard(hash);
            if let Some(ids) = shard.peek(hash, text) {
                #[cfg(feature = "observe")]
                if sample {
                    shard.sample(ids.is_some(), text.len());
                }
                return ids;
            }
        }
        let mut shard = self.write_shard(hash);
        let (ids, dropped) = shard.get(hash, text, self.young_budget);
        #[cfg(feature = "observe")]
        if sample {
            shard.sample(ids.is_some(), text.len());
        }
        // Free a dropped generation without holding the lock.
        drop(shard);
        drop(dropped);
        ids
    }

    /// Cache `ids` as the encoding of `text`, whose [`Self::key`] is `hash`.
    /// A segment costing more than half a generation's budget (1/64 of the
    /// cache's) is not cached: it would evict everything else by itself.
    fn insert(&self, hash: u64, text: &[u8], ids: &[u32]) {
        // Check the cost before copying the text and ids.
        if cost_of(text.len(), ids.len()) > self.young_budget / 2 {
            #[cfg(feature = "observe")]
            {
                self.obs.rejected.fetch_add(1, Relaxed);
                self.obs
                    .rejected_bytes
                    .fetch_add(text.len() as u64, Relaxed);
            }
            return;
        }
        let entry = Entry {
            text: text.into(),
            ids: ids.into(),
        };
        let mut shard = self.write_shard(hash);
        shard.remove_old(hash);
        let dropped = shard.insert_young(hash, entry, self.young_budget);
        #[cfg(feature = "observe")]
        {
            shard.obs.inserts += 1;
        }
        drop(shard);
        drop(dropped);
    }

    /// Count `ns` spent tokenizing a segment the lookup missed.
    #[inline]
    fn record_miss_tokenize(&self, ns: u64) {
        #[cfg(feature = "observe")]
        self.obs.miss_tokenize_ns.fetch_add(ns, Relaxed);
        let _ = ns;
    }

    /// The cache's size and, with the `observe` feature, its counters and
    /// every shard's generations (see [`ShardObs`]), with the hit and miss
    /// size histograms summed over shards.
    pub(crate) fn observe_json(&self) -> serde_json::Value {
        let (mut entries, mut bytes) = (0usize, 0usize);
        #[cfg(feature = "observe")]
        let mut shards: Vec<serde_json::Value> = Vec::with_capacity(SHARDS);
        #[cfg(feature = "observe")]
        let (mut hit_sizes, mut miss_sizes) = (
            [0u64; crate::observe::BUCKETS],
            [0u64; crate::observe::BUCKETS],
        );
        #[cfg(feature = "observe")]
        let (mut hits, mut misses, mut hit_bytes, mut miss_bytes) = (0u64, 0u64, 0u64, 0u64);
        for shard in self.shards.iter() {
            let sh = shard.read().unwrap();
            entries += sh.young.len() + sh.old.len();
            bytes += sh.young_bytes + sh.old_bytes;
            #[cfg(feature = "observe")]
            {
                let o = &sh.obs;
                for b in 0..crate::observe::BUCKETS {
                    hit_sizes[b] += o.hit_sizes[b].load(Relaxed);
                    miss_sizes[b] += o.miss_sizes[b].load(Relaxed);
                }
                hits += sh.hits.load(Relaxed);
                misses += sh.misses.load(Relaxed);
                hit_bytes += sh.hit_bytes.load(Relaxed);
                miss_bytes += sh.miss_bytes.load(Relaxed);
                shards.push(serde_json::json!({
                    "young_entries": sh.young.len(), "young_bytes": sh.young_bytes,
                    "old_entries": sh.old.len(), "old_bytes": sh.old_bytes,
                    "collisions": o.collisions.load(Relaxed), "promotions": o.promotions,
                    "inserts": o.inserts, "replaced": o.replaced,
                    "rotations": o.rotations,
                    "evicted_entries": o.evicted_entries, "evicted_bytes": o.evicted_bytes,
                    "lock_contended": o.contended.load(Relaxed),
                    "lock_wait_ns": o.wait_ns.load(Relaxed),
                }));
            }
        }
        #[allow(unused_mut)]
        let mut v = serde_json::json!({
            // Tokenizers sharing this cache (a copy and its original) report
            // the same id; one given a fresh cache by `add_tokens` a new one.
            "id": self as *const Self as usize,
            "approx_max_bytes": self.young_budget * SHARDS * 2,
            "young_budget_per_shard": self.young_budget,
            "max_segment_cost": self.young_budget / 2,
            "shard_count": SHARDS,
            "entries": entries,
            "bytes": bytes,
        });
        #[cfg(feature = "observe")]
        {
            let o = &self.obs;
            v["shards"] = shards.into();
            v["size_sample_every"] = SIZE_SAMPLE.into();
            v["hit_sizes_log2"] = hit_sizes.to_vec().into();
            v["miss_sizes_log2"] = miss_sizes.to_vec().into();
            v["hits"] = hits.into();
            v["misses"] = misses.into();
            v["hit_bytes"] = hit_bytes.into();
            v["miss_bytes"] = miss_bytes.into();
            for (k, c) in [
                ("rejected_too_large", &o.rejected),
                ("rejected_bytes", &o.rejected_bytes),
                ("miss_tokenize_ns", &o.miss_tokenize_ns),
            ] {
                v[k] = c.load(Relaxed).into();
            }
        }
        v
    }

    pub(crate) fn stats(&self) -> SegmentCacheStats {
        let mut stats = SegmentCacheStats {
            max_bytes: self.max_bytes,
            ..SegmentCacheStats::default()
        };
        for shard in self.shards.iter() {
            let shard = shard.read().unwrap();
            stats.hits += shard.hits.load(Relaxed);
            stats.misses += shard.misses.load(Relaxed);
            stats.hit_bytes += shard.hit_bytes.load(Relaxed);
            stats.miss_bytes += shard.miss_bytes.load(Relaxed);
            stats.entries += shard.young.len() + shard.old.len();
            stats.bytes += shard.young_bytes + shard.old_bytes;
        }
        stats
    }
}

/// Build a [`SegmentCache`] from the `FASTOKENS_SEGMENT_CACHE` env var (a size
/// in MiB), or `None` if it is unset or zero, the default.
pub(crate) fn from_env() -> Option<Arc<SegmentCache>> {
    std::env::var("FASTOKENS_SEGMENT_CACHE")
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .filter(|&mib| mib >= 1)
        .map(|mib| Arc::new(SegmentCache::new(mib.saturating_mul(1 << 20))))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids_of(cache: &SegmentCache, text: &[u8]) -> Option<Arc<[u32]>> {
        cache.get(cache.key(text), text)
    }

    fn put(cache: &SegmentCache, text: &[u8], ids: &[u32]) {
        cache.insert(cache.key(text), text, ids);
    }

    #[test]
    fn keys_are_random_per_cache() {
        let (a, b) = (SegmentCache::new(1 << 20), SegmentCache::new(1 << 20));
        let text = b"<|im_start|>user\nhello<|im_end|>";
        assert_eq!(a.key(text), a.key(text));
        assert_ne!(a.key(text), b.key(text));
    }

    #[test]
    fn hash_depends_on_every_byte_and_the_length() {
        let base = b"<|im_start|>user\nread src/lib.rs and fix the bug<|im_end|>".to_vec();
        let h = hash_bytes(&base);
        for i in 0..base.len() {
            let mut changed = base.clone();
            changed[i] ^= 1;
            assert_ne!(hash_bytes(&changed), h, "byte {i}");
        }
        assert_ne!(hash_bytes(&base[..base.len() - 1]), h);
        assert_ne!(hash_bytes(b""), hash_bytes(b"\0"));
    }

    #[test]
    fn encode_tokenizes_once_and_counts() {
        let cache = SegmentCache::new(1 << 20);
        let mut calls = 0;
        let mut tokenize = |out: &mut Vec<u32>| {
            calls += 1;
            out.extend([1, 2, 3]);
            Ok::<(), ()>(())
        };
        let mut out = vec![9];
        cache
            .encode("hello world", &mut out, &mut tokenize)
            .unwrap();
        cache
            .encode("hello world", &mut out, &mut tokenize)
            .unwrap();
        assert_eq!(out, [9, 1, 2, 3, 1, 2, 3]);
        assert_eq!(calls, 1);
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses), (1, 1));
        assert_eq!((stats.hit_bytes, stats.miss_bytes), (11, 11));
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.bytes, 11 + 12 + ENTRY_OVERHEAD);
        assert_eq!(stats.max_bytes, 1 << 20);
    }

    #[test]
    fn a_hash_collision_is_a_miss_not_a_wrong_answer() {
        let cache = SegmentCache::new(1 << 20);
        cache.insert(42, b"first", &[1]);
        assert!(cache.get(42, b"second").is_none());
        cache.insert(42, b"second", &[2]);
        assert_eq!(&*cache.get(42, b"second").unwrap(), &[2]);
        assert!(cache.get(42, b"first").is_none());
    }

    #[test]
    fn stays_within_budget_and_keeps_recently_used_entries() {
        let budget = 64 * 1024;
        let cache = SegmentCache::new(budget);
        let hot = b"the system prompt every request repeats";
        put(&cache, hot, &[7; 10]);
        // Small entries, and entries as large as the cache accepts.
        let large = vec![b'x'; budget / 64 - ENTRY_OVERHEAD - 4 * 8];
        for i in 0..10_000u32 {
            let text = format!("tool output number {i}");
            put(&cache, text.as_bytes(), &[i; 8]);
            if i % 3 == 0 {
                let mut large = large.clone();
                large[..4].copy_from_slice(&i.to_le_bytes());
                put(&cache, &large, &[i; 8]);
            }
            // Touching the hot entry keeps promoting it out of the old generation.
            assert!(ids_of(&cache, hot).is_some(), "hot entry evicted at {i}");
            let stats = cache.stats();
            assert!(stats.bytes <= budget, "{} bytes at {i}", stats.bytes);
        }
        assert!(cache.stats().entries > 0);
    }

    #[test]
    fn oversized_segments_are_not_cached() {
        let cache = SegmentCache::new(SHARDS * 2 * 1024);
        let big = vec![b'x'; 4096];
        put(&cache, &big, &[1; 100]);
        assert!(ids_of(&cache, &big).is_none());
        assert_eq!(cache.stats().entries, 0);
        // A generation holds 1024 bytes, and an entry at most half of that.
        assert!(cache.can_cache(512 - ENTRY_OVERHEAD));
        assert!(!cache.can_cache(512 - ENTRY_OVERHEAD + 1));
    }

    /// A young hit and a miss go on while another thread reads the shard; an
    /// old hit, which moves the entry, waits for it.
    #[test]
    fn lookups_share_a_shard_but_moving_an_entry_waits() {
        let young = 1000;
        let cache = Arc::new(SegmentCache::new(young * SHARDS * 2));
        let key = |i: u64| (5u64 << 32) | i; // all in shard 5
        let text = |i: u64| format!("entry {i:04}").into_bytes();
        // Nine entries of 122 bytes: the ninth turns the first eight old.
        for i in 0..9 {
            cache.insert(key(i), &text(i), &[i as u32; 4]);
        }
        let reader = cache.shard(key(0)).read().unwrap();
        let (tx, rx) = std::sync::mpsc::channel();
        let c = cache.clone();
        let t = std::thread::spawn(move || {
            let young_hit = c.get(key(8), &text(8));
            let miss = c.get(key(99), &text(99));
            tx.send((young_hit, miss)).unwrap();
            let old_hit = c.get(key(0), &text(0));
            tx.send((old_hit, None)).unwrap();
        });
        let timeout = std::time::Duration::from_secs(10);
        let (young_hit, miss) = rx.recv_timeout(timeout).expect("lookups blocked");
        assert_eq!(young_hit.as_deref(), Some(&[8u32; 4][..]));
        assert!(miss.is_none());
        std::thread::sleep(std::time::Duration::from_millis(50));
        assert!(
            rx.try_recv().is_err(),
            "an old hit moved the entry under a read lock"
        );
        drop(reader);
        let (old_hit, _) = rx.recv_timeout(timeout).unwrap();
        assert_eq!(old_hit.as_deref(), Some(&[0u32; 4][..]));
        t.join().unwrap();
        let stats = cache.stats();
        assert_eq!((stats.hits, stats.misses), (2, 1));
    }

    /// Threads looking up, inserting and moving entries of one small cache at
    /// once (every shard rotates many times) get the right ids, and every
    /// lookup is counted once.
    #[test]
    fn concurrent_lookups_inserts_and_moves_stay_exact() {
        let budget = SHARDS * 2 * 4096;
        let cache = Arc::new(SegmentCache::new(budget));
        let ids_for =
            |k: u64| -> Vec<u32> { (0..(k % 7 + 1) as u32).map(|i| k as u32 * 8 + i).collect() };
        let text_for = |k: u64| format!("{k}:{}", "z".repeat(k as usize % 200)).into_bytes();
        let threads = 8u64;
        let per = 20_000u64;
        let totals: Vec<(u64, u64)> = (0..threads)
            .map(|t| {
                let c = cache.clone();
                std::thread::spawn(move || {
                    let (mut lookups, mut looked_bytes) = (0u64, 0u64);
                    let mut x = 0x9E37_79B9_7F4A_7C15u64 ^ (t + 1);
                    for _ in 0..per {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        // A hot set every thread keeps hitting, and a long tail.
                        let k = if x % 3 == 0 { x % 600 } else { x % 40 };
                        let text = text_for(k);
                        lookups += 1;
                        looked_bytes += text.len() as u64;
                        match ids_of(&c, &text) {
                            Some(ids) => assert_eq!(&*ids, &ids_for(k)[..], "key {k}"),
                            None => put(&c, &text, &ids_for(k)),
                        }
                    }
                    (lookups, looked_bytes)
                })
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|h| h.join().unwrap())
            .collect();
        let stats = cache.stats();
        let lookups: u64 = totals.iter().map(|t| t.0).sum();
        let looked_bytes: u64 = totals.iter().map(|t| t.1).sum();
        assert_eq!(stats.hits + stats.misses, lookups);
        assert_eq!(stats.hit_bytes + stats.miss_bytes, looked_bytes);
        assert!(stats.hits > stats.misses, "{stats:?}");
        assert!(stats.bytes <= budget, "{stats:?}");
        for shard in cache.shards.iter() {
            let sh = shard.read().unwrap();
            let young: usize = sh.young.values().map(Entry::cost).sum();
            let old: usize = sh.old.values().map(Entry::cost).sum();
            assert_eq!((sh.young_bytes, sh.old_bytes), (young, old));
        }
    }

    #[test]
    fn observe_json_reports_size_in_every_build() {
        let cache = SegmentCache::new(SHARDS * 2 * 4096);
        let v = cache.observe_json();
        assert_eq!(v["entries"], 0);
        assert_eq!(v["bytes"], 0);
        assert_eq!(v["shard_count"], SHARDS);
        assert_eq!(v["young_budget_per_shard"], 4096);
        assert_eq!(v["max_segment_cost"], 2048);
        assert_eq!(v["approx_max_bytes"], SHARDS * 2 * 4096);
        put(&cache, b"abc", &[1, 2]);
        let v = cache.observe_json();
        assert_eq!(v["entries"], 1);
        assert_eq!(v["bytes"], 3 + 8 + ENTRY_OVERHEAD);
    }

    #[cfg(feature = "observe")]
    mod observe {
        use super::*;

        fn n(v: &serde_json::Value, k: &str) -> u64 {
            v[k].as_u64().unwrap_or_else(|| panic!("{k} missing"))
        }

        fn shard_sum(v: &serde_json::Value, k: &str) -> u64 {
            v["shards"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| n(s, k))
                .sum()
        }

        /// The old generation's tracked size equals the sum of its entries'
        /// costs, in every shard, after every operation.
        fn check_old_bytes(cache: &SegmentCache) {
            for (i, shard) in cache.shards.iter().enumerate() {
                let sh = shard.read().unwrap();
                let real: usize = sh.old.values().map(Entry::cost).sum();
                assert_eq!(sh.old_bytes, real, "shard {i}");
            }
        }

        #[test]
        fn counts_hits_misses_inserts_and_bytes_exactly() {
            let cache = SegmentCache::new(1 << 20);
            let (a, b) = (b"segment a".as_slice(), b"segment bb".as_slice());
            assert!(ids_of(&cache, a).is_none());
            put(&cache, a, &[1, 2]);
            for _ in 0..3 {
                assert!(ids_of(&cache, a).is_some());
            }
            assert!(ids_of(&cache, b).is_none());
            let v = cache.observe_json();
            assert_eq!((n(&v, "hits"), n(&v, "misses")), (3, 2));
            assert_eq!((n(&v, "hit_bytes"), n(&v, "miss_bytes")), (27, 19));
            assert_eq!(shard_sum(&v, "inserts"), 1);
            assert_eq!(shard_sum(&v, "rotations"), 0);
            assert_eq!(n(&v, "entries"), 1);
            assert_eq!(v["shards"].as_array().unwrap().len(), SHARDS);
        }

        #[test]
        fn rotations_evictions_and_promotions_add_up() {
            // One shard's young budget fits a handful of entries; fill one
            // shard (fixed hash high bits) so the arithmetic is exact.
            let young = 1000;
            let cache = SegmentCache::new(young * SHARDS * 2);
            let shard_of = |i: u64| (5u64 << 32) | i; // all in shard 5
            let text = |i: u64| format!("entry {i:04}").into_bytes(); // 10 bytes
            let cost = (10 + 4 * 4 + ENTRY_OVERHEAD) as u64;
            let fit = young as u64 / cost; // entries a young generation holds
            // The insert that would overfill young turns it old first.
            for i in 0..=fit {
                cache.insert(shard_of(i), &text(i), &[0; 4]);
                check_old_bytes(&cache);
            }
            let v = cache.observe_json();
            assert_eq!(shard_sum(&v, "rotations"), 1);
            assert_eq!(
                shard_sum(&v, "evicted_entries"),
                0,
                "the first rotation drops an empty old generation"
            );
            let s5 = &v["shards"][5];
            assert_eq!(
                (n(s5, "old_entries"), n(s5, "old_bytes")),
                (fit, fit * cost)
            );
            assert_eq!((n(s5, "young_entries"), n(s5, "young_bytes")), (1, cost));
            // A hit in the old generation moves the entry back to young.
            assert!(cache.get(shard_of(3), &text(3)).is_some());
            check_old_bytes(&cache);
            let v = cache.observe_json();
            assert_eq!(shard_sum(&v, "promotions"), 1);
            assert_eq!(n(&v["shards"][5], "old_entries"), fit - 1);
            // Young holds two; filling it and overfilling it once more evicts
            // what is left in old.
            let more = fit - 1;
            for i in 100..100 + more {
                cache.insert(shard_of(i), &text(i), &[0; 4]);
                check_old_bytes(&cache);
            }
            let v = cache.observe_json();
            assert_eq!(shard_sum(&v, "rotations"), 2);
            assert_eq!(shard_sum(&v, "evicted_entries"), fit - 1);
            assert_eq!(shard_sum(&v, "evicted_bytes"), (fit - 1) * cost);
            assert!(cache.get(shard_of(0), &text(0)).is_none(), "evicted");
            // A collision: same hash, other text (101 is in old).
            assert!(cache.get(shard_of(101), b"something else").is_none());
            assert_eq!(shard_sum(&cache.observe_json(), "collisions"), 1);
            // The last insert is young: re-inserting its hash replaces it ...
            let last = 100 + more - 1;
            cache.insert(shard_of(last), &text(last), &[1; 4]);
            assert_eq!(shard_sum(&cache.observe_json(), "replaced"), 1);
            // ... and re-inserting an old one moves it to young.
            let old_before = n(&cache.observe_json()["shards"][5], "old_entries");
            cache.insert(shard_of(101), &text(101), &[1; 4]);
            let v = cache.observe_json();
            assert_eq!(shard_sum(&v, "replaced"), 1);
            assert_eq!(n(&v["shards"][5], "old_entries"), old_before - 1);
            check_old_bytes(&cache);
        }

        #[test]
        fn old_bytes_stay_exact_under_random_use() {
            let budget = SHARDS * 2 * 2048;
            let cache = SegmentCache::new(budget);
            let mut x = 0x2545_F491_4F6C_DD1Du64;
            for step in 0..20_000 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let k = x % 400;
                let text = format!("{k}:{}", "y".repeat((x >> 20) as usize % 300));
                if x % 4 == 2 {
                    put(&cache, text.as_bytes(), &[k as u32; 3]);
                } else {
                    ids_of(&cache, text.as_bytes());
                }
                if step % 50 == 0 {
                    check_old_bytes(&cache);
                }
            }
            check_old_bytes(&cache);
            let v = cache.observe_json();
            assert!(shard_sum(&v, "rotations") > 0);
            assert!(n(&v, "bytes") <= budget as u64);
        }

        #[test]
        fn too_large_segments_are_counted_as_rejected() {
            let cache = SegmentCache::new(SHARDS * 2 * 1024);
            let big = vec![b'x'; 4096];
            put(&cache, &big, &[1; 100]);
            let v = cache.observe_json();
            assert_eq!(
                (n(&v, "rejected_too_large"), n(&v, "rejected_bytes")),
                (1, 4096)
            );
            assert_eq!(shard_sum(&v, "inserts"), 0);
        }

        #[test]
        fn a_held_shard_lock_is_counted_as_contention() {
            let cache = Arc::new(SegmentCache::new(1 << 20));
            let text = b"contended";
            let h = cache.key(text);
            let guard = cache.shard(h).write().unwrap();
            let c = cache.clone();
            let t = std::thread::spawn(move || c.get(h, text).is_none());
            std::thread::sleep(std::time::Duration::from_millis(30));
            drop(guard);
            assert!(t.join().unwrap());
            let v = cache.observe_json();
            assert_eq!(shard_sum(&v, "lock_contended"), 1);
            assert!(shard_sum(&v, "lock_wait_ns") >= 20_000_000);
            // Uncontended lookups add nothing.
            cache.get(h, text);
            assert_eq!(shard_sum(&cache.observe_json(), "lock_contended"), 1);
        }

        #[test]
        fn size_histograms_sample_one_lookup_in_sixteen() {
            std::thread::spawn(|| {
                let cache = SegmentCache::new(1 << 20);
                let text = vec![b'z'; 300]; // bucket 9: [256, 512)
                put(&cache, &text, &[1]);
                for _ in 0..1600 {
                    ids_of(&cache, &text);
                }
                let v = cache.observe_json();
                let hits: u64 = v["hit_sizes_log2"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_u64().unwrap())
                    .sum();
                assert_eq!(hits, 100);
                assert_eq!(v["hit_sizes_log2"][9], 100);
                assert_eq!(v["size_sample_every"], SIZE_SAMPLE);
            })
            .join()
            .unwrap();
        }

        #[test]
        fn miss_tokenize_time_is_counted() {
            let cache = SegmentCache::new(1 << 20);
            let mut out = Vec::new();
            let slow = |out: &mut Vec<u32>| {
                std::thread::sleep(std::time::Duration::from_millis(2));
                out.push(1);
                Ok::<(), ()>(())
            };
            cache.encode("a segment", &mut out, slow).unwrap();
            cache.encode("a segment", &mut out, slow).unwrap();
            let v = cache.observe_json();
            assert_eq!((n(&v, "hits"), n(&v, "misses")), (1, 1));
            assert!(n(&v, "miss_tokenize_ns") >= 2_000_000);
        }

        #[test]
        fn concurrent_use_keeps_totals_consistent() {
            let cache = Arc::new(SegmentCache::new(1 << 20));
            let threads = 8;
            let per = 2_000u64;
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let c = cache.clone();
                    std::thread::spawn(move || {
                        for i in 0..per {
                            let text = format!("seg {}", (i * 7 + t) % 64);
                            if ids_of(&c, text.as_bytes()).is_none() {
                                put(&c, text.as_bytes(), &[1, 2, 3]);
                            }
                        }
                    })
                })
                .collect();
            for h in hs {
                h.join().unwrap();
            }
            let v = cache.observe_json();
            assert_eq!(n(&v, "hits") + n(&v, "misses"), threads * per);
            assert_eq!(shard_sum(&v, "inserts"), n(&v, "misses"));
            assert_eq!(n(&v, "entries"), 64);
            check_old_bytes(&cache);
        }
    }
}
