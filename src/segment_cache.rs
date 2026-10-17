//! Opt-in cache of the token ids of text segments between added tokens.
//!
//! Chat-template output, the input a serving stack encodes, is text segments
//! separated by added tokens (`<|im_start|>`, `<|im_end|>`, tool-call
//! markers, …). An agent re-sends its whole history on every request, so all
//! but the newest segments were already encoded by an earlier request.
//!
//! A segment's ids depend on nothing but its normalized text: added tokens are
//! hard boundaries, normalizers run per segment, and the supported
//! pre-tokenizers (`Split`, `ByteLevel` and sequences of them) and the model
//! act on each segment independently. So ids can be reused by content, which
//! also survives history rewrites (dropped reasoning, compaction) that break a
//! byte-prefix cache. Every hit is confirmed against the stored text, so the
//! result is identical to encoding from scratch.
//!
//! The cache is sharded, with a byte budget per shard, and evicts with two
//! generations: entries are inserted into a young map; when that fills up, the
//! old map is dropped and the young one becomes old. A hit on an old entry
//! moves it back into the young map, so recently used segments survive.

#[cfg(feature = "observe")]
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::{
    collections::HashMap,
    hash::{BuildHasherDefault, Hasher},
    sync::{Arc, Mutex},
};

/// Number of independently locked shards.
const SHARDS: usize = 16;

/// Bookkeeping bytes charged per entry on top of its text and ids.
const ENTRY_OVERHEAD: usize = 64;

/// 64-bit hash of `bytes`.
///
/// Four multiply-rotate lanes over 32-byte blocks keep the loop from being a
/// single dependency chain, and a murmur3 finalizer mixes every output bit.
/// It is not collision-resistant and need not be: a hit is confirmed by
/// comparing the stored text, so a collision only costs a cache miss.
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
        self.text.len() + self.ids.len() * 4 + ENTRY_OVERHEAD
    }
}

#[derive(Default)]
struct Shard {
    young: HashMapU64<Entry>,
    old: HashMapU64<Entry>,
    young_bytes: usize,
    #[cfg(feature = "observe")]
    obs: ShardObs,
}

/// A shard's [`crate::observe`] counters, updated under its lock.
#[cfg(feature = "observe")]
#[derive(Default)]
struct ShardObs {
    /// Bytes of the old generation (the cache itself only tracks the young).
    old_bytes: usize,
    /// Lookups whose hash matched an entry of different text.
    collisions: u64,
    /// Old-generation hits moved back into the young generation.
    promotions: u64,
    inserts: u64,
    /// Inserts that replaced an entry with the same hash.
    replaced: u64,
    /// Young generations turned old, each dropping the previous old one.
    rotations: u64,
    evicted_entries: u64,
    evicted_bytes: u64,
    /// Lock acquisitions that had to wait, and how long they waited.
    contended: u64,
    wait_ns: u64,
    /// Segment sizes of every [`SIZE_SAMPLE`]th lookup of a thread: counting
    /// every lookup inside the lock lengthens the critical section enough to
    /// show when many threads share the cache.
    hit_sizes: [u64; crate::observe::BUCKETS],
    miss_sizes: [u64; crate::observe::BUCKETS],
}

/// One lookup in this many per thread lands in the size histograms.
#[cfg(feature = "observe")]
const SIZE_SAMPLE: u32 = 16;

#[cfg(feature = "observe")]
thread_local! {
    static LOOKUPS: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Cache-wide [`crate::observe`] counters (lookups happen outside any one
/// shard's bookkeeping, and misses are tokenized outside the cache).
#[cfg(feature = "observe")]
#[derive(Default)]
struct CacheObs {
    hits: AtomicU64,
    misses: AtomicU64,
    hit_bytes: AtomicU64,
    miss_bytes: AtomicU64,
    rejected: AtomicU64,
    rejected_bytes: AtomicU64,
    miss_tokenize_ns: AtomicU64,
}

impl Shard {
    /// Insert into the young generation, turning it old once it is full.
    fn insert_young(&mut self, hash: u64, entry: Entry, budget: usize) {
        self.young_bytes += entry.cost();
        if let Some(replaced) = self.young.insert(hash, entry) {
            self.young_bytes -= replaced.cost();
            #[cfg(feature = "observe")]
            {
                self.obs.replaced += 1;
            }
        }
        if self.young_bytes > budget {
            #[cfg(feature = "observe")]
            {
                self.obs.rotations += 1;
                self.obs.evicted_entries += self.old.len() as u64;
                self.obs.evicted_bytes += self.obs.old_bytes as u64;
                self.obs.old_bytes = self.young_bytes;
            }
            self.old = std::mem::take(&mut self.young);
            self.young_bytes = 0;
        }
    }

    /// Remove `hash` from the old generation.
    fn remove_old(&mut self, hash: u64) -> Option<Entry> {
        let entry = self.old.remove(&hash)?;
        #[cfg(feature = "observe")]
        {
            self.obs.old_bytes -= entry.cost();
        }
        Some(entry)
    }
}

/// Content-addressed cache of text-segment encodings. See the module docs.
pub(crate) struct SegmentCache {
    shards: Box<[Mutex<Shard>]>,
    /// Byte budget of each shard's young generation. With the old generation
    /// at most as large, the cache stays within the budget it was built with.
    young_budget: usize,
    #[cfg(feature = "observe")]
    obs: CacheObs,
}

impl SegmentCache {
    /// A cache holding at most about `max_bytes` of segment text and ids.
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| Mutex::default()).collect(),
            young_budget: (max_bytes / SHARDS / 2).max(1),
            #[cfg(feature = "observe")]
            obs: CacheObs::default(),
        }
    }

    fn shard(&self, hash: u64) -> &Mutex<Shard> {
        // Bits the maps' bucket index (low) and tag (top) do not use.
        &self.shards[(hash >> 32) as usize % SHARDS]
    }

    /// Lock the shard of `hash`, counting a wait if another thread held it.
    fn lock_shard(&self, hash: u64) -> std::sync::MutexGuard<'_, Shard> {
        let m = self.shard(hash);
        #[cfg(feature = "observe")]
        match m.try_lock() {
            Ok(shard) => return shard,
            Err(std::sync::TryLockError::WouldBlock) => {
                let t = std::time::Instant::now();
                let mut shard = m.lock().unwrap();
                shard.obs.contended += 1;
                shard.obs.wait_ns += t.elapsed().as_nanos() as u64;
                return shard;
            }
            // Poisoned: the plain lock below panics as it always did.
            Err(std::sync::TryLockError::Poisoned(_)) => {}
        }
        m.lock().unwrap()
    }

    /// The cached ids of `text`, whose [`hash_bytes`] is `hash`.
    pub(crate) fn get(&self, hash: u64, text: &[u8]) -> Option<Arc<[u32]>> {
        #[cfg(feature = "observe")]
        let sample = LOOKUPS
            .try_with(|n| {
                let k = n.get().wrapping_add(1);
                n.set(k);
                k % SIZE_SAMPLE == 0
            })
            .unwrap_or(false);
        let mut shard = self.lock_shard(hash);
        let found = Self::get_locked(&mut shard, hash, text, self.young_budget);
        #[cfg(feature = "observe")]
        {
            if sample {
                let b = crate::observe::bucket(text.len() as u64);
                match found {
                    Some(_) => shard.obs.hit_sizes[b] += 1,
                    None => shard.obs.miss_sizes[b] += 1,
                }
            }
            drop(shard);
            let (n, bytes) = match found {
                Some(_) => (&self.obs.hits, &self.obs.hit_bytes),
                None => (&self.obs.misses, &self.obs.miss_bytes),
            };
            n.fetch_add(1, Relaxed);
            bytes.fetch_add(text.len() as u64, Relaxed);
        }
        found
    }

    fn get_locked(shard: &mut Shard, hash: u64, text: &[u8], budget: usize) -> Option<Arc<[u32]>> {
        if let Some(entry) = shard.young.get(&hash) {
            let hit = *entry.text == *text;
            let ids = hit.then(|| entry.ids.clone());
            #[cfg(feature = "observe")]
            if !hit {
                shard.obs.collisions += 1;
            }
            return ids;
        }
        match shard.old.get(&hash) {
            None => return None,
            Some(entry) if *entry.text != *text => {
                #[cfg(feature = "observe")]
                {
                    shard.obs.collisions += 1;
                }
                return None;
            }
            Some(_) => {}
        }
        let entry = shard.remove_old(hash).unwrap();
        let ids = entry.ids.clone();
        shard.insert_young(hash, entry, budget);
        #[cfg(feature = "observe")]
        {
            shard.obs.promotions += 1;
        }
        Some(ids)
    }

    /// Cache `ids` as the encoding of `text`, whose [`hash_bytes`] is `hash`.
    /// A segment too large for its shard is not cached.
    pub(crate) fn insert(&self, hash: u64, text: &[u8], ids: &[u32]) {
        let entry = Entry {
            text: text.into(),
            ids: ids.into(),
        };
        if entry.cost() > self.young_budget / 2 {
            #[cfg(feature = "observe")]
            {
                self.obs.rejected.fetch_add(1, Relaxed);
                self.obs
                    .rejected_bytes
                    .fetch_add(text.len() as u64, Relaxed);
            }
            return;
        }
        let mut shard = self.lock_shard(hash);
        shard.remove_old(hash);
        shard.insert_young(hash, entry, self.young_budget);
        #[cfg(feature = "observe")]
        {
            shard.obs.inserts += 1;
        }
    }

    /// Count `ns` spent tokenizing the segments a lookup missed.
    #[inline]
    pub(crate) fn record_miss_tokenize(&self, ns: u64) {
        #[cfg(feature = "observe")]
        self.obs.miss_tokenize_ns.fetch_add(ns, Relaxed);
        let _ = ns;
    }

    /// Drop every entry (the [`crate::observe`] counters keep counting).
    pub(crate) fn clear(&self) {
        for shard in self.shards.iter() {
            let mut shard = shard.lock().unwrap();
            #[cfg(feature = "observe")]
            let mut obs = std::mem::take(&mut shard.obs);
            *shard = Shard::default();
            #[cfg(feature = "observe")]
            {
                obs.old_bytes = 0;
                shard.obs = obs;
            }
        }
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
        for shard in self.shards.iter() {
            let sh = shard.lock().unwrap();
            let old_bytes: usize = sh.old.values().map(Entry::cost).sum();
            entries += sh.young.len() + sh.old.len();
            bytes += sh.young_bytes + old_bytes;
            #[cfg(feature = "observe")]
            {
                let o = &sh.obs;
                for b in 0..crate::observe::BUCKETS {
                    hit_sizes[b] += o.hit_sizes[b];
                    miss_sizes[b] += o.miss_sizes[b];
                }
                shards.push(serde_json::json!({
                    "young_entries": sh.young.len(), "young_bytes": sh.young_bytes,
                    "old_entries": sh.old.len(), "old_bytes": old_bytes,
                    "collisions": o.collisions, "promotions": o.promotions,
                    "inserts": o.inserts, "replaced": o.replaced,
                    "rotations": o.rotations,
                    "evicted_entries": o.evicted_entries, "evicted_bytes": o.evicted_bytes,
                    "lock_contended": o.contended, "lock_wait_ns": o.wait_ns,
                }));
            }
        }
        #[allow(unused_mut)]
        let mut v = serde_json::json!({
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
            for (k, c) in [
                ("hits", &o.hits),
                ("misses", &o.misses),
                ("hit_bytes", &o.hit_bytes),
                ("miss_bytes", &o.miss_bytes),
                ("rejected_too_large", &o.rejected),
                ("rejected_bytes", &o.rejected_bytes),
                ("miss_tokenize_ns", &o.miss_tokenize_ns),
            ] {
                v[k] = c.load(Relaxed).into();
            }
        }
        v
    }

    #[cfg(test)]
    fn len(&self) -> usize {
        self.shards
            .iter()
            .map(|s| {
                let s = s.lock().unwrap();
                s.young.len() + s.old.len()
            })
            .sum()
    }
}

/// Build a [`SegmentCache`] from the `FASTOKENS_SEGMENT_CACHE` env var (a size
/// in MiB), or `None` if it is unset or zero, the default.
pub(crate) fn from_env() -> Option<SegmentCache> {
    std::env::var("FASTOKENS_SEGMENT_CACHE")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .filter(|&mib| mib >= 1)
        .map(|mib| SegmentCache::new(mib.saturating_mul(1 << 20)))
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn get_returns_what_was_inserted() {
        let cache = SegmentCache::new(1 << 20);
        let text = b"hello world";
        let h = hash_bytes(text);
        assert!(cache.get(h, text).is_none());
        cache.insert(h, text, &[1, 2, 3]);
        assert_eq!(&*cache.get(h, text).unwrap(), &[1, 2, 3]);
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
        let hot = b"the system prompt every request repeats".to_vec();
        let hot_hash = hash_bytes(&hot);
        cache.insert(hot_hash, &hot, &[7; 10]);
        for i in 0..10_000u32 {
            let text = format!("tool output number {i}");
            cache.insert(hash_bytes(text.as_bytes()), text.as_bytes(), &[i; 8]);
            // Touching the hot entry keeps promoting it out of the old generation.
            assert!(
                cache.get(hot_hash, &hot).is_some(),
                "hot entry evicted at {i}"
            );
        }
        let max_entries = budget / (ENTRY_OVERHEAD + 20 + 32);
        assert!(cache.len() <= max_entries, "{} entries", cache.len());
    }

    #[test]
    fn oversized_segments_are_not_cached() {
        let cache = SegmentCache::new(SHARDS * 2 * 1024);
        let big = vec![b'x'; 4096];
        let h = hash_bytes(&big);
        cache.insert(h, &big, &[1; 100]);
        assert!(cache.get(h, &big).is_none());
    }

    #[test]
    fn clear_drops_everything() {
        let cache = SegmentCache::new(1 << 20);
        cache.insert(hash_bytes(b"a"), b"a", &[1]);
        cache.clear();
        assert!(cache.get(hash_bytes(b"a"), b"a").is_none());
        assert_eq!(cache.len(), 0);
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
        cache.insert(hash_bytes(b"abc"), b"abc", &[1, 2]);
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
                let sh = shard.lock().unwrap();
                let real: usize = sh.old.values().map(Entry::cost).sum();
                assert_eq!(sh.obs.old_bytes, real, "shard {i}");
            }
        }

        #[test]
        fn counts_hits_misses_inserts_and_bytes_exactly() {
            let cache = SegmentCache::new(1 << 20);
            let (a, b) = (b"segment a".as_slice(), b"segment bb".as_slice());
            assert!(cache.get(hash_bytes(a), a).is_none());
            cache.insert(hash_bytes(a), a, &[1, 2]);
            for _ in 0..3 {
                assert!(cache.get(hash_bytes(a), a).is_some());
            }
            assert!(cache.get(hash_bytes(b), b).is_none());
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
            // cost = 10 + 4 * 4 + 64 = 90: the 12th insert pushes young over 1000
            for i in 0..12 {
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
            assert_eq!((n(s5, "old_entries"), n(s5, "old_bytes")), (12, 12 * 90));
            assert_eq!((n(s5, "young_entries"), n(s5, "young_bytes")), (0, 0));
            // A hit in the old generation moves the entry back to young.
            assert!(cache.get(shard_of(3), &text(3)).is_some());
            check_old_bytes(&cache);
            let v = cache.observe_json();
            assert_eq!(shard_sum(&v, "promotions"), 1);
            assert_eq!(n(&v["shards"][5], "old_entries"), 11);
            // Twelve more: a second rotation evicts the 11 left in old.
            for i in 100..112 {
                cache.insert(shard_of(i), &text(i), &[0; 4]);
                check_old_bytes(&cache);
            }
            let v = cache.observe_json();
            assert_eq!(shard_sum(&v, "rotations"), 2);
            assert_eq!(shard_sum(&v, "evicted_entries"), 11);
            assert_eq!(shard_sum(&v, "evicted_bytes"), 11 * 90);
            assert!(cache.get(shard_of(0), &text(0)).is_none(), "evicted");
            // A collision: same hash, other text.
            assert!(cache.get(shard_of(101), b"something else").is_none());
            assert_eq!(shard_sum(&cache.observe_json(), "collisions"), 1);
            // The second rotation came at the 11th of those inserts: 3 and
            // 100..=110 are old, 111 is young. Re-inserting a young entry's
            // hash replaces it ...
            cache.insert(shard_of(111), &text(111), &[1; 4]);
            assert_eq!(shard_sum(&cache.observe_json(), "replaced"), 1);
            // ... and re-inserting an old one moves it to young.
            let old_before = n(&cache.observe_json()["shards"][5], "old_entries");
            cache.insert(shard_of(105), &text(105), &[1; 4]);
            let v = cache.observe_json();
            assert_eq!(shard_sum(&v, "replaced"), 1);
            assert_eq!(n(&v["shards"][5], "old_entries"), old_before - 1);
            check_old_bytes(&cache);
        }

        #[test]
        fn old_bytes_stay_exact_under_random_use() {
            let cache = SegmentCache::new(SHARDS * 2 * 2048);
            let mut x = 0x2545_F491_4F6C_DD1Du64;
            for step in 0..20_000 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                let k = x % 400;
                let text = format!("{k}:{}", "y".repeat((x >> 20) as usize % 300));
                let h = hash_bytes(text.as_bytes());
                match x % 4 {
                    0 | 1 => {
                        cache.get(h, text.as_bytes());
                    }
                    2 => cache.insert(h, text.as_bytes(), &[k as u32; 3]),
                    _ if step % 997 == 0 => cache.clear(),
                    _ => {
                        cache.get(h, text.as_bytes());
                    }
                }
                if step % 50 == 0 {
                    check_old_bytes(&cache);
                }
            }
            check_old_bytes(&cache);
            let v = cache.observe_json();
            assert!(shard_sum(&v, "rotations") > 0);
            assert!(n(&v, "bytes") <= (SHARDS * 2 * 2048 + SHARDS * 2048) as u64);
        }

        #[test]
        fn too_large_segments_are_counted_as_rejected() {
            let cache = SegmentCache::new(SHARDS * 2 * 1024);
            let big = vec![b'x'; 4096];
            cache.insert(hash_bytes(&big), &big, &[1; 100]);
            let v = cache.observe_json();
            assert_eq!(
                (n(&v, "rejected_too_large"), n(&v, "rejected_bytes")),
                (1, 4096)
            );
            assert_eq!(shard_sum(&v, "inserts"), 0);
        }

        #[test]
        fn a_held_shard_lock_is_counted_as_contention() {
            let cache = std::sync::Arc::new(SegmentCache::new(1 << 20));
            let text = b"contended";
            let h = hash_bytes(text);
            let guard = cache.shard(h).lock().unwrap();
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
                let h = hash_bytes(&text);
                cache.insert(h, &text, &[1]);
                for _ in 0..1600 {
                    cache.get(h, &text);
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
        fn clear_keeps_counters_and_resets_sizes() {
            let cache = SegmentCache::new(1 << 20);
            cache.insert(hash_bytes(b"a"), b"a", &[1]);
            cache.get(hash_bytes(b"a"), b"a");
            cache.clear();
            check_old_bytes(&cache);
            let v = cache.observe_json();
            assert_eq!(n(&v, "hits"), 1);
            assert_eq!(shard_sum(&v, "inserts"), 1);
            assert_eq!((n(&v, "entries"), n(&v, "bytes")), (0, 0));
        }

        #[test]
        fn concurrent_use_keeps_totals_consistent() {
            let cache = std::sync::Arc::new(SegmentCache::new(1 << 20));
            let threads = 8;
            let per = 2_000u64;
            let hs: Vec<_> = (0..threads)
                .map(|t| {
                    let c = cache.clone();
                    std::thread::spawn(move || {
                        for i in 0..per {
                            let text = format!("seg {}", (i * 7 + t) % 64);
                            let h = hash_bytes(text.as_bytes());
                            if c.get(h, text.as_bytes()).is_none() {
                                c.insert(h, text.as_bytes(), &[1, 2, 3]);
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
