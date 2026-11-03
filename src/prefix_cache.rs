//! Prefix cache for input that contains added tokens: chat-template output,
//! the input an inference server encodes.
//!
//! A chat API is stateless, so every request re-sends the whole conversation,
//! and the rendered prompt of the next request usually begins with the bytes
//! of the previous one. The cache keeps the encodings of recent inputs and,
//! for a new input, reuses the ids of the longest byte prefix it shares with
//! one of them; only the rest is encoded.
//!
//! Reuse is cut where an added token starts (a [`Cut`]). Added tokens are
//! matched on the raw input before anything else runs, and everything after
//! the match handles the text between two tokens on its own, so the ids
//! before an added token do not depend on what follows it — provided the new
//! input still has that token there. The shared bytes guarantee it, except
//! that a token can only be matched once enough bytes follow its start: a
//! longer token (matching is leftmost-longest) may start at or just before
//! the cut and run past the shared bytes. [`choose_cut`] checks those few
//! positions in the new input, so the result is identical to encoding from
//! scratch.
//!
//! Inputs are found by the chained hash of each full [`BLOCK`] of their
//! bytes, which costs one pass over the input however many inputs are cached.
//! The candidate with the deepest matching block is then compared byte by
//! byte, so a hash collision only costs reuse.
//!
//! An input that continues the conversation of the entry it reused replaces
//! it, so the next request finds the newest one and a conversation keeps one
//! entry (see [`Hit::continued_by_input`]). Inputs of other conversations that
//! only share a system prompt with an entry leave it alone.
//!
//! The cache holds at most `capacity` entries and `max_bytes` bytes, evicting
//! the least recently used; an input whose entry would cost more than 1/64 of
//! `max_bytes` is not cached.

use std::{
    collections::{BTreeMap, HashMap},
    sync::{
        Arc, Mutex, MutexGuard,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
};

use crate::segment_cache::{HashMapU64, hash_bytes};

/// Inputs are indexed by the chained hash of each full block of this many
/// bytes; an input must share at least one with a cached input to reuse it.
pub(crate) const BLOCK: usize = 4096;

/// Bookkeeping bytes charged per entry on top of its arrays.
const ENTRY_OVERHEAD: usize = 128;

/// An entry may cost at most this fraction of the byte budget, as in the
/// segment cache: one huge input cannot flush the cache.
const MAX_ENTRY_SHARE: usize = 64;

/// Where a cached input can be cut: an added token's span starts at `start`
/// (its text at `content`, ending at `content_end`), and the input before
/// `start` encodes to the entry's first `ids_before` ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Cut {
    pub start: u32,
    pub content: u32,
    pub content_end: u32,
    pub ids_before: u32,
}

/// One cached input with its core ids (before post-processing).
pub(crate) struct Entry {
    id: u64,
    pub bytes: Box<[u8]>,
    pub ids: Box<[u32]>,
    /// Ascending by `start`.
    pub cuts: Box<[Cut]>,
    chain: Box<[u64]>,
    /// Leading bytes this input shares with other conversations, as far as
    /// the cache could tell when it was inserted: the prefix it reused from
    /// another conversation's entry, or its whole length when nothing
    /// matched (it may be the shared prompt itself). See
    /// [`Hit::continued_by_input`].
    shared: usize,
}

impl Entry {
    fn cost(&self) -> usize {
        self.bytes.len()
            + self.ids.len() * 4
            + self.cuts.len() * std::mem::size_of::<Cut>()
            + self.chain.len() * 8
            + ENTRY_OVERHEAD
    }
}

/// A cached input sharing the first `lcp` bytes of the input looked up.
pub(crate) struct Hit {
    pub entry: Arc<Entry>,
    pub lcp: usize,
}

impl Hit {
    /// The entry's id, to refresh it on an exact hit.
    pub fn entry_id(&self) -> u64 {
        self.entry.id
    }

    /// Whether the input looked up continues the entry's conversation, so
    /// that it replaces the entry.
    ///
    /// It does if it contains the whole entry: then any later input shares at
    /// least as much with it as with the entry. Otherwise (a history rewrite
    /// such as dropped reasoning, or another conversation) it must share at
    /// least half of what the entry does not share with other conversations.
    /// Measuring that half from the start instead lets every conversation
    /// whose history is still shorter than twice a shared system prompt
    /// replace the others' entries.
    pub fn continued_by_input(&self) -> bool {
        let (len, shared) = (self.entry.bytes.len(), self.entry.shared);
        self.lcp >= len || (self.lcp > shared && 2 * (self.lcp - shared) >= len - shared)
    }
}

struct Inner {
    /// id -> (entry, last-use tick).
    entries: HashMap<u64, (Arc<Entry>, u64)>,
    /// last-use tick -> id, oldest first.
    lru: BTreeMap<u64, u64>,
    /// Chained block hash -> id of the newest entry with that block chain.
    index: HashMapU64<u64>,
    tick: u64,
    bytes: usize,
}

impl Inner {
    fn touch(&mut self, id: u64) {
        self.tick += 1;
        let tick = self.tick;
        if let Some((_, t)) = self.entries.get_mut(&id) {
            self.lru.remove(t);
            *t = tick;
            self.lru.insert(tick, id);
        }
    }

    /// Drop entry `id`, returning the bytes it held (`None` if it was gone).
    fn remove(&mut self, id: u64) -> Option<usize> {
        let (entry, tick) = self.entries.remove(&id)?;
        self.lru.remove(&tick);
        self.bytes -= entry.cost();
        for h in entry.chain.iter() {
            if self.index.get(h) == Some(&id) {
                self.index.remove(h);
            }
        }
        Some(entry.cost())
    }
}

/// The prefix cache. See the module docs.
pub(crate) struct PrefixCache {
    capacity: usize,
    max_bytes: usize,
    next_id: AtomicU64,
    inner: Mutex<Inner>,
    #[cfg(feature = "observe")]
    obs: Counters,
}

/// What a lookup reused, for [`PrefixCache::count`].
#[derive(Clone, Copy)]
pub(crate) enum Outcome {
    /// The whole encoding of an identical input.
    Exact,
    /// The ids of a shared prefix.
    Prefix,
    Miss,
}

/// Something the inference path did besides a lookup, for [`PrefixCache::note`].
#[derive(Clone, Copy)]
pub(crate) enum Event {
    /// A cached input shared a whole added token, but a later cut was ruled
    /// out (a token running past the shared bytes, or one starting in absorbed
    /// whitespace), so reuse stopped at an earlier one.
    CutBackoff,
    /// A cached input shared a whole added token, but no cut could be used.
    NoCut,
    /// Input shorter than one [`BLOCK`]: not looked up.
    BypassShort,
    /// Input without added tokens, found so after its lookup: left to the
    /// plain-text prefix cache (on the scanner path, which counts its own
    /// lookup); not counted as a lookup here.
    BypassPlain,
    /// A model id equal to the stand-in id: encoded without the cache.
    MarkerFallback,
}

/// [`observe`](crate::observe) counters; [`PrefixCache::observe_json`] lists them.
#[cfg(feature = "observe")]
#[derive(Default)]
struct Counters {
    lookups: AtomicU64,
    exact: AtomicU64,
    prefix: AtomicU64,
    lookup_bytes: AtomicU64,
    reused_bytes: AtomicU64,
    reused_tokens: AtomicU64,
    encoded_bytes: AtomicU64,
    cut_backoffs: AtomicU64,
    no_cut: AtomicU64,
    bypass_short: AtomicU64,
    bypass_plain: AtomicU64,
    marker_fallbacks: AtomicU64,
    inserts: AtomicU64,
    replaced: AtomicU64,
    evicted_entries: AtomicU64,
    evicted_bytes: AtomicU64,
    rejected_oversized: AtomicU64,
    rejected_bytes: AtomicU64,
    cleared_entries: AtomicU64,
    lock_contended: AtomicU64,
    lock_wait_ns: AtomicU64,
}

#[cfg(feature = "observe")]
fn add(counter: &AtomicU64, n: u64) {
    counter.fetch_add(n, Relaxed);
}

impl PrefixCache {
    /// A cache holding up to `capacity` inputs and about `max_bytes` bytes
    /// (inputs, ids and bookkeeping).
    pub fn new(capacity: usize, max_bytes: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            max_bytes,
            next_id: AtomicU64::new(0),
            inner: Mutex::new(Inner {
                entries: HashMap::new(),
                lru: BTreeMap::new(),
                index: HashMapU64::default(),
                tick: 0,
                bytes: 0,
            }),
            #[cfg(feature = "observe")]
            obs: Counters::default(),
        }
    }

    /// Lock the cache, counting a wait if another thread held it.
    fn lock(&self) -> MutexGuard<'_, Inner> {
        #[cfg(feature = "observe")]
        match self.inner.try_lock() {
            Ok(inner) => return inner,
            Err(std::sync::TryLockError::WouldBlock) => {
                let t = std::time::Instant::now();
                let inner = self.inner.lock().unwrap();
                add(&self.obs.lock_contended, 1);
                add(&self.obs.lock_wait_ns, t.elapsed().as_nanos() as u64);
                return inner;
            }
            // Poisoned: the plain lock below panics as it always did.
            Err(std::sync::TryLockError::Poisoned(_)) => {}
        }
        self.inner.lock().unwrap()
    }

    /// Count a lookup of an `input_len`-byte input for [`observe`](crate::observe):
    /// its outcome, and the bytes and tokens it reused (the rest was encoded).
    #[inline]
    pub fn count(
        &self,
        outcome: Outcome,
        input_len: usize,
        reused_bytes: usize,
        reused_tokens: usize,
    ) {
        #[cfg(feature = "observe")]
        {
            let o = &self.obs;
            add(&o.lookups, 1);
            add(&o.lookup_bytes, input_len as u64);
            match outcome {
                Outcome::Exact => add(&o.exact, 1),
                Outcome::Prefix => add(&o.prefix, 1),
                Outcome::Miss => {}
            }
            add(&o.reused_bytes, reused_bytes as u64);
            add(&o.reused_tokens, reused_tokens as u64);
            add(&o.encoded_bytes, (input_len - reused_bytes) as u64);
        }
        #[cfg(not(feature = "observe"))]
        let _ = (outcome, input_len, reused_bytes, reused_tokens);
    }

    /// Count `event` for [`observe`](crate::observe).
    #[inline]
    pub fn note(&self, event: Event) {
        #[cfg(feature = "observe")]
        {
            let o = &self.obs;
            add(
                match event {
                    Event::CutBackoff => &o.cut_backoffs,
                    Event::NoCut => &o.no_cut,
                    Event::BypassShort => &o.bypass_short,
                    Event::BypassPlain => &o.bypass_plain,
                    Event::MarkerFallback => &o.marker_fallbacks,
                },
                1,
            );
        }
        #[cfg(not(feature = "observe"))]
        let _ = event;
    }

    /// The cache's size (entries, bytes it charges itself, index blocks) and,
    /// with the `observe` feature, its counters. The keys the plain-text
    /// prefix cache also has mean the same here.
    pub fn observe_json(&self) -> serde_json::Value {
        #[allow(unused_mut)]
        let mut v = {
            let inner = self.lock();
            serde_json::json!({
                "capacity": self.capacity,
                "max_bytes": self.max_bytes,
                "entries": inner.entries.len(),
                "bytes": inner.bytes,
                "index_blocks": inner.index.len(),
            })
        };
        #[cfg(feature = "observe")]
        {
            let o = &self.obs;
            let (lookups, exact, prefix) = (
                o.lookups.load(Relaxed),
                o.exact.load(Relaxed),
                o.prefix.load(Relaxed),
            );
            v["lookups"] = lookups.into();
            v["exact_hits"] = exact.into();
            v["prefix_hits"] = prefix.into();
            v["misses"] = (lookups - exact - prefix).into();
            for (key, counter) in [
                ("lookup_bytes", &o.lookup_bytes),
                ("reused_bytes", &o.reused_bytes),
                ("reused_tokens", &o.reused_tokens),
                ("encoded_bytes", &o.encoded_bytes),
                ("cut_backoffs", &o.cut_backoffs),
                ("no_cut", &o.no_cut),
                ("bypass_short", &o.bypass_short),
                ("bypass_plain", &o.bypass_plain),
                ("marker_fallbacks", &o.marker_fallbacks),
                ("inserts", &o.inserts),
                ("replaced", &o.replaced),
                ("evicted_entries", &o.evicted_entries),
                ("evicted_bytes", &o.evicted_bytes),
                ("rejected_oversized", &o.rejected_oversized),
                ("rejected_bytes", &o.rejected_bytes),
                ("cleared_entries", &o.cleared_entries),
                ("lock_contended", &o.lock_contended),
                ("lock_wait_ns", &o.lock_wait_ns),
            ] {
                v[key] = counter.load(Relaxed).into();
            }
        }
        v
    }

    /// Drop every entry. Required whenever the vocabulary or the normalizer
    /// changes.
    pub fn clear(&self) {
        let mut inner = self.lock();
        #[cfg(feature = "observe")]
        add(&self.obs.cleared_entries, inner.entries.len() as u64);
        inner.entries.clear();
        inner.lru.clear();
        inner.index.clear();
        inner.bytes = 0;
    }

    /// Bytes held: inputs, ids and bookkeeping.
    #[cfg(test)]
    pub fn bytes(&self) -> usize {
        self.lock().bytes
    }

    /// Number of cached inputs.
    #[cfg(test)]
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    /// Panic unless the bookkeeping agrees with the entries and the limits.
    #[cfg(test)]
    pub fn check_invariants(&self) {
        let inner = self.lock();
        assert_eq!(inner.lru.len(), inner.entries.len());
        for (tick, id) in &inner.lru {
            assert_eq!(inner.entries[id].1, *tick);
        }
        let cost: usize = inner.entries.values().map(|(e, _)| e.cost()).sum();
        assert_eq!(inner.bytes, cost);
        assert!(inner.entries.len() <= self.capacity);
        assert!(inner.bytes <= self.max_bytes);
        for id in inner.index.values() {
            assert!(
                inner.entries.contains_key(id),
                "index points to a dropped entry"
            );
        }
    }

    /// The chained hash of each full [`BLOCK`] of `bytes`. `seed` keeps inputs
    /// encoded under different added-token policies apart.
    pub fn chain(bytes: &[u8], seed: u64) -> Vec<u64> {
        let mut h = seed;
        bytes
            .as_chunks::<BLOCK>()
            .0
            .iter()
            .map(|block| {
                let mut link = [0u8; 16];
                link[..8].copy_from_slice(&h.to_le_bytes());
                link[8..].copy_from_slice(&hash_bytes(block).to_le_bytes());
                h = hash_bytes(&link);
                h
            })
            .collect()
    }

    /// The cached input sharing the longest byte prefix with `bytes`, found by
    /// its deepest block in `chain` (from [`Self::chain`]).
    pub fn lookup(&self, bytes: &[u8], chain: &[u64]) -> Option<Hit> {
        let entry = {
            let inner = self.lock();
            let id = chain.iter().rev().find_map(|h| inner.index.get(h))?;
            inner.entries.get(id)?.0.clone()
        };
        let lcp = common_prefix_len(&entry.bytes, bytes);
        Some(Hit { entry, lcp })
    }

    /// Mark entry `id` as just used.
    pub fn touch(&self, id: u64) {
        self.lock().touch(id);
    }

    /// Cache the encoding of `bytes`, which reused `matched` (the result of
    /// its lookup): replace that entry if this input continues it, or mark it
    /// as used. Then evict the least recently used inputs beyond capacity or
    /// budget.
    pub fn insert(
        &self,
        bytes: &[u8],
        ids: Box<[u32]>,
        cuts: Box<[Cut]>,
        chain: Vec<u64>,
        matched: Option<&Hit>,
    ) {
        let continues = matched.filter(|h| h.continued_by_input());
        let shared = match (matched, continues) {
            (_, Some(h)) => h.entry.shared,
            (Some(h), None) => h.lcp,
            (None, None) => bytes.len(),
        };
        // Built before taking the lock: copying the input is the slow part.
        let entry = Entry {
            id: self.next_id.fetch_add(1, Relaxed),
            bytes: bytes.into(),
            ids,
            cuts,
            chain: chain.into_boxed_slice(),
            shared,
        };
        let cost = entry.cost();
        if cost > self.max_bytes / MAX_ENTRY_SHARE {
            // Keep what the input reused: it is still the best prefix there is.
            if let Some(h) = matched {
                self.touch(h.entry_id());
            }
            #[cfg(feature = "observe")]
            {
                add(&self.obs.rejected_oversized, 1);
                add(&self.obs.rejected_bytes, cost as u64);
            }
            return;
        }
        let entry = Arc::new(entry);
        let id = entry.id;

        let mut inner = self.lock();
        let replaced = match (matched, continues) {
            (_, Some(h)) => inner.remove(h.entry_id()).is_some(),
            (Some(h), None) => {
                inner.touch(h.entry_id());
                false
            }
            (None, None) => false,
        };
        for &h in entry.chain.iter() {
            inner.index.insert(h, id);
        }
        inner.bytes += cost;
        inner.tick += 1;
        let tick = inner.tick;
        inner.lru.insert(tick, id);
        inner.entries.insert(id, (entry, tick));
        let (mut evicted, mut evicted_bytes) = (0u64, 0usize);
        // The new entry is the most recent and within budget on its own, so
        // it is never the one evicted.
        while inner.entries.len() > self.capacity || inner.bytes > self.max_bytes {
            let (_, &oldest) = inner.lru.first_key_value().expect("lru tracks every entry");
            evicted_bytes += inner.remove(oldest).unwrap_or(0);
            evicted += 1;
        }
        #[cfg(feature = "observe")]
        {
            let o = &self.obs;
            add(&o.inserts, 1);
            add(&o.replaced, u64::from(replaced));
            add(&o.evicted_entries, evicted);
            add(&o.evicted_bytes, evicted_bytes as u64);
        }
        #[cfg(not(feature = "observe"))]
        let _ = (replaced, evicted, evicted_bytes);
    }
}

/// Longest common prefix of `a` and `b`: block by block with `memcmp`, then
/// byte by byte inside the first block that differs.
fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + BLOCK <= n && a[i..i + BLOCK] == b[i..i + BLOCK] {
        i += BLOCK;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// The index of the last cut of `cuts` whose token lies inside the first `lcp`
/// bytes, which [`choose_cut`] takes unless a later check rules it out.
pub(crate) fn last_shared_cut(cuts: &[Cut], lcp: usize) -> Option<usize> {
    cuts.partition_point(|c| c.content_end as usize <= lcp)
        .checked_sub(1)
}

/// The index of the last cut of `cuts` at which an input sharing the first
/// `lcp` bytes of the cached input can be cut, or `None`.
///
/// The token at the cut must lie inside the shared bytes, and no added token
/// may match the new input at a position up to that token's start and run
/// past `lcp` (`ends_after_lcp(pos)`): such a match would change how the new
/// input splits there. Only the last `max_token_len` positions before `lcp`
/// can start one.
pub(crate) fn choose_cut(
    cuts: &[Cut],
    lcp: usize,
    max_token_len: usize,
    ends_after_lcp: impl Fn(usize) -> bool,
) -> Option<usize> {
    let shared = cuts.partition_point(|c| c.content_end as usize <= lcp);
    let last = cuts[..shared].last()?;
    let lo = (lcp + 1).saturating_sub(max_token_len);
    let limit = (lo..=last.content as usize)
        .find(|&pos| ends_after_lcp(pos))
        .unwrap_or(usize::MAX);
    cuts[..shared]
        .partition_point(|c| (c.content as usize) < limit)
        .checked_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cut(start: u32, len: u32, ids_before: u32) -> Cut {
        Cut {
            start,
            content: start,
            content_end: start + len,
            ids_before,
        }
    }

    #[test]
    fn choose_cut_takes_the_last_token_inside_the_shared_bytes() {
        let cuts = [cut(10, 5, 3), cut(40, 5, 9), cut(90, 5, 20)];
        let never = |_| false;
        assert_eq!(choose_cut(&cuts, 9, 8, never), None);
        assert_eq!(choose_cut(&cuts, 15, 8, never), Some(0));
        assert_eq!(choose_cut(&cuts, 44, 8, never), Some(0));
        assert_eq!(choose_cut(&cuts, 45, 8, never), Some(1));
        assert_eq!(choose_cut(&cuts, 1000, 8, never), Some(2));
    }

    #[test]
    fn choose_cut_backs_off_before_a_token_running_past_the_shared_bytes() {
        let cuts = [cut(10, 5, 3), cut(40, 5, 9)];
        // A match at 38 that runs past lcp = 45 rules out the token at 40.
        assert_eq!(choose_cut(&cuts, 45, 8, |pos| pos == 38), Some(0));
        // So does a longer token at the cut itself.
        assert_eq!(choose_cut(&cuts, 45, 8, |pos| pos == 40), Some(0));
        // Positions before lcp + 1 - max_token_len cannot reach past lcp and
        // are not probed.
        assert_eq!(choose_cut(&cuts, 45, 8, |pos| pos < 38), Some(1));
    }

    const NO_BUDGET: usize = 1 << 40;

    #[test]
    fn lookup_finds_the_entry_with_the_deepest_shared_block() {
        let cache = PrefixCache::new(8, NO_BUDGET);
        let base: Vec<u8> = (0..3 * BLOCK).map(|i| (i % 251) as u8).collect();
        let mut a = base.clone();
        a.extend(std::iter::repeat_n(b'a', 2 * BLOCK));
        let mut b = base.clone();
        b.extend(std::iter::repeat_n(b'b', 2 * BLOCK));
        for (bytes, n) in [(&a, 1u32), (&b, 2)] {
            let chain = PrefixCache::chain(bytes, 0);
            cache.insert(bytes, vec![n].into(), Box::new([]), chain, None);
        }
        let mut query = a.clone();
        query.extend_from_slice(b"tail");
        let hit = cache
            .lookup(&query, &PrefixCache::chain(&query, 0))
            .unwrap();
        assert_eq!(&*hit.entry.ids, &[1]);
        assert_eq!(hit.lcp, a.len());

        // Another policy seed shares no block.
        assert!(
            cache
                .lookup(&query, &PrefixCache::chain(&query, 1))
                .is_none()
        );
    }

    #[cfg(feature = "observe")]
    #[test]
    fn a_held_lock_is_counted_as_contention() {
        let cache = Arc::new(PrefixCache::new(4, NO_BUDGET));
        let bytes = vec![b'q'; 2 * BLOCK];
        let chain = PrefixCache::chain(&bytes, 0);
        let guard = cache.inner.lock().unwrap();
        let (c, b, ch) = (cache.clone(), bytes.clone(), chain.clone());
        let t = std::thread::spawn(move || c.lookup(&b, &ch).is_none());
        std::thread::sleep(std::time::Duration::from_millis(30));
        drop(guard);
        assert!(t.join().unwrap());
        let v = cache.observe_json();
        assert_eq!(v["lock_contended"], 1);
        assert!(v["lock_wait_ns"].as_u64().unwrap() >= 20_000_000);
        // Uncontended calls add nothing.
        cache.lookup(&bytes, &chain);
        assert_eq!(cache.observe_json()["lock_contended"], 1);
    }

    #[test]
    fn insert_replaces_and_evicts_least_recently_used() {
        let cache = PrefixCache::new(2, NO_BUDGET);
        let input = |c: u8| vec![c; 2 * BLOCK];
        let add = |bytes: &[u8], matched| {
            let chain = PrefixCache::chain(bytes, 0);
            cache.insert(bytes, vec![0; 10].into(), Box::new([]), chain, matched);
        };
        add(&input(b'a'), None);
        add(&input(b'b'), None);
        let hit_a = cache
            .lookup(&input(b'a'), &PrefixCache::chain(&input(b'a'), 0))
            .unwrap();
        cache.touch(hit_a.entry_id());
        add(&input(b'c'), None); // evicts b, the least recently used
        assert_eq!(cache.len(), 2);
        assert!(
            cache
                .lookup(&input(b'b'), &PrefixCache::chain(&input(b'b'), 0))
                .is_none()
        );
        assert!(
            cache
                .lookup(&input(b'a'), &PrefixCache::chain(&input(b'a'), 0))
                .is_some()
        );

        // An input containing all of `a` replaces it.
        add(&input(b'd'), Some(&hit_a));
        assert_eq!(cache.len(), 2);
        assert!(
            cache
                .lookup(&input(b'a'), &PrefixCache::chain(&input(b'a'), 0))
                .is_none()
        );

        cache.clear();
        assert_eq!((cache.len(), cache.bytes()), (0, 0));
        cache.check_invariants();
    }

    /// `blocks` full blocks: `shared` of a common prompt, then `c`.
    fn convo(shared: usize, c: u8, blocks: usize) -> Vec<u8> {
        let mut v: Vec<u8> = (0..shared * BLOCK).map(|i| (i % 251) as u8).collect();
        v.resize(blocks * BLOCK, c);
        v
    }

    /// Look `bytes` up and cache it as the encoder does; the entry it reused.
    fn encode(cache: &PrefixCache, bytes: &[u8]) -> Option<Hit> {
        let chain = PrefixCache::chain(bytes, 0);
        let hit = cache.lookup(bytes, &chain);
        cache.insert(bytes, vec![1; 10].into(), Box::new([]), chain, hit.as_ref());
        hit
    }

    #[test]
    fn conversations_sharing_a_long_prompt_keep_their_own_entries() {
        let cache = PrefixCache::new(16, NO_BUDGET);
        // A 3-block prompt, longer than what follows it: under a plain "shares
        // half of the entry" rule each conversation would replace the last.
        assert!(encode(&cache, &convo(3, b'a', 4)).is_none());
        let b = encode(&cache, &convo(3, b'b', 4)).unwrap();
        assert_eq!(b.lcp, 3 * BLOCK);
        assert!(!b.continued_by_input());
        encode(&cache, &convo(3, b'c', 4));
        assert_eq!(cache.len(), 3);

        // Each conversation's next request finds and replaces its own entry.
        for c in *b"abc" {
            let hit = encode(&cache, &convo(3, c, 6)).unwrap();
            assert_eq!(hit.lcp, 4 * BLOCK, "{}", c as char);
            assert!(hit.continued_by_input());
        }
        assert_eq!(cache.len(), 3);

        // A rewrite keeping more than half of what is the conversation's own
        // (blocks 3..6 of `b`) continues it; one keeping less does not.
        let mut rewrite = convo(3, b'b', 5);
        rewrite.resize(7 * BLOCK, b'x');
        assert!(encode(&cache, &rewrite).unwrap().continued_by_input());
        assert_eq!(cache.len(), 3);
        let mut early = convo(3, b'b', 4);
        early.resize(6 * BLOCK, b'y');
        let hit = encode(&cache, &early).unwrap();
        assert_eq!(hit.lcp, 4 * BLOCK);
        assert!(!hit.continued_by_input());
        assert_eq!(cache.len(), 4);
        cache.check_invariants();
    }

    #[test]
    fn the_byte_budget_evicts_least_recently_used() {
        let input = |i: usize| vec![i as u8; 2 * BLOCK];
        let cost = {
            let one = PrefixCache::new(1, NO_BUDGET);
            encode(&one, &input(0));
            one.bytes()
        };
        // 64 entries fit; capacity is not the limit.
        let cache = PrefixCache::new(1000, 64 * cost + cost / 2);
        for i in 0..64 {
            encode(&cache, &input(i));
        }
        assert_eq!(cache.len(), 64);
        let first = &input(0);
        let hit = cache.lookup(first, &PrefixCache::chain(first, 0)).unwrap();
        cache.touch(hit.entry_id());
        for i in 64..70 {
            encode(&cache, &input(i));
        }
        assert_eq!(cache.len(), 64);
        assert!(cache.bytes() <= 64 * cost + cost / 2);
        let found = |i: usize| {
            cache
                .lookup(&input(i), &PrefixCache::chain(&input(i), 0))
                .is_some()
        };
        assert!(found(0), "touched, so kept");
        assert!(
            (1..7).all(|i| !found(i)),
            "the six least recently used are gone"
        );
        assert!((7..70).all(found));
        cache.check_invariants();
        #[cfg(feature = "observe")]
        {
            let v = cache.observe_json();
            assert_eq!(v["evicted_entries"], 6);
            assert_eq!(v["evicted_bytes"], 6 * cost);
            assert_eq!(v["max_bytes"], 64 * cost + cost / 2);
        }
    }

    #[test]
    fn an_input_costing_over_a_64th_of_the_budget_is_not_cached() {
        let small = convo(0, b'a', 2);
        let cost = {
            let one = PrefixCache::new(1, NO_BUDGET);
            encode(&one, &small);
            one.bytes()
        };
        let cache = PrefixCache::new(8, 64 * (cost + 100));
        assert!(encode(&cache, &small).is_none());
        assert_eq!(cache.len(), 1);
        // Twice the size: rejected, and the entry it continues stays.
        let big = convo(0, b'a', 4);
        let hit = encode(&cache, &big).unwrap();
        assert!(hit.continued_by_input());
        assert_eq!((cache.len(), cache.bytes()), (1, cost));
        assert!(
            cache
                .lookup(&small, &PrefixCache::chain(&small, 0))
                .is_some()
        );
        cache.check_invariants();
        #[cfg(feature = "observe")]
        {
            let v = cache.observe_json();
            assert_eq!(v["rejected_oversized"], 1);
            assert!(v["rejected_bytes"].as_u64().unwrap() > 2 * cost as u64 - 1000);
            assert_eq!(
                (v["inserts"].as_u64(), v["replaced"].as_u64()),
                (Some(1), Some(0))
            );
        }
    }

    #[test]
    fn concurrent_replacement_and_eviction_keep_the_books() {
        // Two threads continue the same conversation and evict each other's
        // entries through a capacity of one; every lookup must still return
        // an entry whose bytes are what its hit claims.
        let cache = Arc::new(PrefixCache::new(1, NO_BUDGET));
        let threads: Vec<_> = (0..2u8)
            .map(|t| {
                let cache = cache.clone();
                std::thread::spawn(move || {
                    for i in 0..400usize {
                        let mut input = convo(2, b'a', 3 + i % 5);
                        input.resize(input.len() + BLOCK, b'0' + t);
                        let chain = PrefixCache::chain(&input, 0);
                        let hit = cache.lookup(&input, &chain);
                        if let Some(h) = &hit {
                            assert_eq!(h.lcp, common_prefix_len(&h.entry.bytes, &input));
                            assert!(h.lcp >= 2 * BLOCK);
                        }
                        let ids: Box<[u32]> = vec![i as u32; 4].into();
                        cache.insert(&input, ids, Box::new([]), chain, hit.as_ref());
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(cache.len(), 1);
        cache.check_invariants();
    }
}
