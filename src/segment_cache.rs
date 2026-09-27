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
struct PassThrough(u64);

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

type HashMapU64<V> = HashMap<u64, V, BuildHasherDefault<PassThrough>>;

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
}

impl Shard {
    /// Insert into the young generation, turning it old once it is full.
    fn insert_young(&mut self, hash: u64, entry: Entry, budget: usize) {
        self.young_bytes += entry.cost();
        if let Some(replaced) = self.young.insert(hash, entry) {
            self.young_bytes -= replaced.cost();
        }
        if self.young_bytes > budget {
            self.old = std::mem::take(&mut self.young);
            self.young_bytes = 0;
        }
    }
}

/// Content-addressed cache of text-segment encodings. See the module docs.
pub(crate) struct SegmentCache {
    shards: Box<[Mutex<Shard>]>,
    /// Byte budget of each shard's young generation. With the old generation
    /// at most as large, the cache stays within the budget it was built with.
    young_budget: usize,
}

impl SegmentCache {
    /// A cache holding at most about `max_bytes` of segment text and ids.
    pub(crate) fn new(max_bytes: usize) -> Self {
        Self {
            shards: (0..SHARDS).map(|_| Mutex::default()).collect(),
            young_budget: (max_bytes / SHARDS / 2).max(1),
        }
    }

    fn shard(&self, hash: u64) -> &Mutex<Shard> {
        // Bits the maps' bucket index (low) and tag (top) do not use.
        &self.shards[(hash >> 32) as usize % SHARDS]
    }

    /// The cached ids of `text`, whose [`hash_bytes`] is `hash`.
    pub(crate) fn get(&self, hash: u64, text: &[u8]) -> Option<Arc<[u32]>> {
        let mut shard = self.shard(hash).lock().unwrap();
        if let Some(entry) = shard.young.get(&hash) {
            return (*entry.text == *text).then(|| entry.ids.clone());
        }
        if shard
            .old
            .get(&hash)
            .is_none_or(|entry| *entry.text != *text)
        {
            return None;
        }
        let entry = shard.old.remove(&hash).unwrap();
        let ids = entry.ids.clone();
        shard.insert_young(hash, entry, self.young_budget);
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
            return;
        }
        let mut shard = self.shard(hash).lock().unwrap();
        shard.old.remove(&hash);
        shard.insert_young(hash, entry, self.young_budget);
    }

    /// Drop every entry.
    pub(crate) fn clear(&self) {
        for shard in self.shards.iter() {
            *shard.lock().unwrap() = Shard::default();
        }
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
}
