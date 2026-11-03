//! The prefix cache for input with added tokens (chat templates) against the
//! same pipeline without it: cut placement and backoffs, the replacement rule,
//! the byte budget, strict-prefix inputs, threads sharing one entry, the
//! full-scan split, clearing and the scanner path. `tests/prefix_cache_fuzz.rs`
//! fuzzes it against real tokenizers.

use serde_json::{Value, json};

use crate::{
    AddedTokenPolicy, NewToken, Tokenizer,
    added_tokens::{SpannedSegment, TokenSpan},
    normalizers::Normalizer,
    prefix_cache::{self, PrefixCache},
    segment_cache_tests::{SCAN_SPECIALS, scan_test_tokenizer},
    test_support::chat_requests,
    tiktoken,
};

/// Added tokens that make cutting a prefix delicate: `<|a|>>` extends
/// `<|a|>`; `x<|b|>y` starts before `<|b|>` and runs past it; `<|b|>`
/// absorbs whitespace on both sides and `<|c|>` on its left, where
/// ` <|c|>z` can start.
const PREFIX_ADDED: [(&str, bool, bool, bool); 6] = [
    ("<|a|>", false, false, true),
    ("<|a|>>", false, false, true),
    ("<|b|>", true, true, false),
    ("x<|b|>y", false, false, true),
    ("<|c|>", true, false, true),
    (" <|c|>z", false, false, true),
];

/// A byte-level BPE tokenizer with [`PREFIX_ADDED`] (lstrip, rstrip,
/// special) and the given normalizer and pre-tokenizer.
pub(crate) fn prefix_test_tokenizer(normalizer: Value, pre_tokenizer: Value) -> Tokenizer {
    prefix_test_tokenizer_with(&PREFIX_ADDED, normalizer, pre_tokenizer)
}

/// [`prefix_test_tokenizer`] with the added tokens `added`.
fn prefix_test_tokenizer_with(
    added: &[(&str, bool, bool, bool)],
    normalizer: Value,
    pre_tokenizer: Value,
) -> Tokenizer {
    let mut alphabet: Vec<char> = tokenizers::pre_tokenizers::byte_level::ByteLevel::alphabet()
        .into_iter()
        .collect();
    alphabet.sort_unstable();
    let mut vocab: serde_json::Map<String, Value> = alphabet
        .iter()
        .enumerate()
        .map(|(id, c)| (c.to_string(), json!(id)))
        .collect();
    // `e Ġ` and `x Ġ` merge across where `<|c|>` absorbs whitespace, and
    // `Ġ x` across the start of `x<|b|>y`.
    let merges = [
        "Ġ t", "h e", "Ġt he", "i n", "in g", "Ċ Ċ", "Ġ Ġ", "e Ġ", "x Ġ", "Ġ x",
    ];
    for merge in merges {
        let id = vocab.len();
        vocab.insert(merge.replace(' ', ""), json!(id));
    }
    let n = vocab.len();
    let added: Vec<Value> = added
        .iter()
        .enumerate()
        .map(|(i, &(content, lstrip, rstrip, special))| {
            json!({"id": n + i, "content": content, "single_word": false, "lstrip": lstrip,
                   "rstrip": rstrip, "normalized": false, "special": special})
        })
        .collect();
    Tokenizer::from_json(json!({
        "version": "1.0",
        "added_tokens": added,
        "normalizer": normalizer,
        "pre_tokenizer": pre_tokenizer,
        "post_processor": null,
        "decoder": {"type": "ByteLevel"},
        "model": {"type": "BPE", "vocab": vocab, "merges": merges},
    }))
    .unwrap()
}

pub(crate) fn prefix_byte_level() -> Value {
    json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true})
}

/// Text that forms, breaks and extends [`PREFIX_ADDED`] tokens.
const PREFIX_PIECES: [&str; 23] = [
    "<|a|>",
    "<|a|>>",
    "<|b|>",
    "x<|b|>y",
    "<|c|>",
    " <|c|>z",
    "<|a",
    "|>",
    ">",
    "x",
    "y",
    "z",
    "<",
    " ",
    "  ",
    "\n",
    "\n\n",
    "the thing",
    "hello world ",
    "the",
    "é",
    "中文",
    "<|im_end|>",
];

/// A stream of requests that mostly continue an earlier one: each is an
/// earlier request (usually the latest) cut at a random char boundary and
/// extended with [`PREFIX_PIECES`].
fn prefix_test_requests(seed: u64, count: usize) -> Vec<String> {
    prefix_test_requests_from(seed, count, &PREFIX_PIECES)
}

/// [`prefix_test_requests`] extended with `pieces`.
fn prefix_test_requests_from(seed: u64, count: usize, pieces: &[&str]) -> Vec<String> {
    let mut x = seed;
    let mut next = move || {
        x = x
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (x >> 33) as usize
    };
    let mut requests: Vec<String> = Vec::new();
    for _ in 0..count {
        let mut s = match requests.len() {
            0 => String::new(),
            n if next() % 4 == 0 => requests[next() % n].clone(),
            n => requests[n - 1].clone(),
        };
        if !s.is_empty() && next() % 3 != 0 {
            let mut cut = s.len() - next() % s.len().min(64);
            while !s.is_char_boundary(cut) {
                cut -= 1;
            }
            s.truncate(cut);
        }
        let target = s.len().max(prefix_cache::BLOCK) + next() % 3000;
        while s.len() < target {
            s.push_str(pieces[next() % pieces.len()]);
        }
        requests.push(s);
    }
    requests
}

#[test]
fn prefix_cache_matches_uncached_encoding() {
    let byte_level = |regex: bool| json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": regex});
    let configs = [
        ("byte-level regex", Value::Null, byte_level(true)),
        ("bulk byte-level", Value::Null, byte_level(false)),
        (
            "NFC, bulk byte-level",
            json!({"type": "NFC"}),
            byte_level(false),
        ),
        (
            "prepend normalizer",
            json!({"type": "Prepend", "prepend": ">"}),
            byte_level(true),
        ),
    ];
    for (name, normalizer, pre_tokenizer) in configs {
        let plain = prefix_test_tokenizer(normalizer.clone(), pre_tokenizer.clone());
        // The last runs on a budget that evicts by bytes and turns away
        // inputs over about 20 KB.
        for (seed, capacity, segments, budget) in [
            (1, 64, false, None),
            (2, 2, false, None),
            (3, 64, true, None),
            (4, 64, false, Some(64 * 45_000)),
        ] {
            let requests = prefix_test_requests(seed, 80);
            let mut cached = prefix_test_tokenizer(normalizer.clone(), pre_tokenizer.clone());
            match budget {
                Some(max_bytes) => cached.enable_input_cache_with_budget(capacity, max_bytes),
                None => cached.enable_input_cache(capacity),
            }
            if segments {
                cached.enable_segment_cache(1 << 20);
            }
            for (i, request) in requests.iter().enumerate() {
                for policy in [AddedTokenPolicy::All, AddedTokenPolicy::SkipSpecial] {
                    assert_eq!(
                        cached.encode_with_policy(request, true, policy).unwrap(),
                        plain.encode_with_policy(request, true, policy).unwrap(),
                        "{name}: seed {seed}, request {i}, {policy:?}"
                    );
                }
            }
            // Again, mostly from the cache, and concurrently.
            let expected: Vec<Vec<u32>> =
                requests.iter().map(|r| plain.encode(r).unwrap()).collect();
            assert_eq!(
                cached.encode_batch(&requests, false).unwrap(),
                expected,
                "{name}: batch"
            );
            cached.prefix_cache.as_ref().unwrap().check_invariants();
        }
    }
}

#[test]
fn prefix_cache_matches_uncached_encoding_on_the_full_scan_split() {
    // A fourth first byte (`#`) and more distinct 4-byte prefixes than the
    // prefilters take move the added-token split to the full automaton scan.
    let numbered: Vec<String> = (0..64).map(|i| format!("#t{i:02}#")).collect();
    let mut added = PREFIX_ADDED.to_vec();
    added.push(("#t#", false, true, true));
    added.extend(numbered.iter().map(|t| (t.as_str(), false, false, true)));
    let plain = prefix_test_tokenizer_with(&added, Value::Null, prefix_byte_level());
    assert!(plain.added_tokens().unwrap().uses_full_scan());
    let mut pieces = PREFIX_PIECES.to_vec();
    pieces.extend(["#t#", "#t#", "#t", "t#", " #t# ", "#", "#t07#", "#t0", "7#"]);
    for (seed, capacity, segments) in [(11, 64, false), (12, 2, false), (13, 64, true)] {
        let mut cached = prefix_test_tokenizer_with(&added, Value::Null, prefix_byte_level());
        cached.enable_input_cache(capacity);
        if segments {
            cached.enable_segment_cache(1 << 20);
        }
        for (i, request) in prefix_test_requests_from(seed, 80, &pieces)
            .iter()
            .enumerate()
        {
            for policy in [AddedTokenPolicy::All, AddedTokenPolicy::SkipSpecial] {
                assert_eq!(
                    cached.encode_with_policy(request, true, policy).unwrap(),
                    plain.encode_with_policy(request, true, policy).unwrap(),
                    "seed {seed}, request {i}, {policy:?}"
                );
            }
        }
        cached.prefix_cache.as_ref().unwrap().check_invariants();
    }
}

#[test]
fn a_tiny_prefix_cache_budget_is_raised_to_one_mib() {
    // Below 64 bytes the per-entry limit would be zero and turn every
    // input away. At 1 MiB it is 16 KiB: room for a ~4 KB input.
    let mut tok = prefix_test_tokenizer(Value::Null, prefix_byte_level());
    tok.enable_input_cache_with_budget(4, 10);
    tok.encode(&format!("<|a|>{}<|a|>", "the thing ".repeat(420)))
        .unwrap();
    let cache = tok.prefix_cache.as_ref().unwrap();
    assert_eq!(cache.len(), 1);
    assert_eq!(cache.observe_json()["max_bytes"], 1 << 20);
}

#[test]
fn prefix_cache_handles_inputs_that_are_strict_prefixes_of_an_entry() {
    let bulk = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false});
    let plain = prefix_test_tokenizer(Value::Null, bulk.clone());
    let mut cached = prefix_test_tokenizer(Value::Null, bulk);
    cached.enable_input_cache(4);
    let body = "the thing ".repeat(500);
    let full = format!("<|c|>{body}x  <|c|>z<|a|>>{body}<|b|>  x<|b|>y{body}<|a|>");
    // Every end within a few bytes of a token, and some in between.
    let near_token = |i: usize| {
        ["<", ">", "x"]
            .iter()
            .any(|t| full[i.saturating_sub(8)..(i + 8).min(full.len())].contains(t))
    };
    let ends: Vec<usize> = (prefix_cache::BLOCK..full.len())
        .filter(|&i| full.is_char_boundary(i) && (near_token(i) || i % 997 == 0))
        .collect();
    assert!(ends.len() > 50);
    for (k, &end) in ends.iter().enumerate() {
        // A cut-off retry of the full input, then the full input again,
        // and every few ends a shorter input on its own.
        for text in [&full[..end], &full] {
            assert_eq!(
                cached.encode(text).unwrap(),
                plain.encode(text).unwrap(),
                "end {end}"
            );
        }
        if k % 7 == 0 {
            let mut fresh = prefix_test_tokenizer(
                Value::Null,
                json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}),
            );
            fresh.enable_input_cache(4);
            fresh.encode(&full).unwrap();
            assert_eq!(
                fresh.encode(&full[..end]).unwrap(),
                plain.encode(&full[..end]).unwrap()
            );
        }
    }
    cached.prefix_cache.as_ref().unwrap().check_invariants();
}

#[test]
fn prefix_cache_stays_correct_when_threads_replace_and_evict_one_entry() {
    // A capacity of one: two threads continuing conversations that share
    // a prompt keep replacing and evicting the same entry.
    let plain = prefix_test_tokenizer(Value::Null, prefix_byte_level());
    let mut cached = prefix_test_tokenizer(Value::Null, prefix_byte_level());
    cached.enable_input_cache(1);
    let prompt = format!("<|c|>{}<|a|>", "the thing ".repeat(600));
    std::thread::scope(|scope| {
        for t in 0..2 {
            let (cached, plain, prompt) = (&cached, &plain, &prompt);
            scope.spawn(move || {
                let mut text = prompt.clone();
                for i in 0..120 {
                    if i % 15 == 14 {
                        // A rewrite back into the history.
                        text.truncate(prompt.len() + (text.len() - prompt.len()) / 2);
                        while !text.is_char_boundary(text.len()) {
                            text.pop();
                        }
                    }
                    text.push_str(&format!("x {t}.{i} <|b|> the thing<|a|>>"));
                    assert_eq!(
                        cached.encode(&text).unwrap(),
                        plain.encode(&text).unwrap(),
                        "thread {t}, request {i}"
                    );
                }
            });
        }
    });
    let cache = cached.prefix_cache.as_ref().unwrap();
    assert_eq!(cache.len(), 1);
    cache.check_invariants();
}

#[test]
fn caches_are_emptied_and_stay_correct_when_the_normalizer_changes() {
    let prepend = |p: &str| {
        Normalizer::from_config(
            serde_json::from_value(json!({"type": "Prepend", "prepend": p})).unwrap(),
        )
        .unwrap()
    };
    let mut plain = prefix_test_tokenizer(Value::Null, prefix_byte_level());
    let mut cached = prefix_test_tokenizer(Value::Null, prefix_byte_level());
    cached.enable_input_cache(4);
    cached.enable_segment_cache(1 << 20);
    let chat = format!("<|a|>{}<|c|>the thing<|a|>", "say hi to ".repeat(500));
    let short = "<|a|>say hi<|c|>the thing<|a|>";
    let before: Vec<Vec<u32>> = [&chat[..], short]
        .iter()
        .map(|t| cached.encode(t).unwrap())
        .collect();
    let segments = |t: &Tokenizer| {
        t.observe_json()["segment_cache"]["entries"]
            .as_u64()
            .unwrap()
    };
    assert!(cached.prefix_cache.as_ref().unwrap().len() > 0 && segments(&cached) > 0);
    for tok in [&mut plain, &mut cached] {
        tok.set_normalizer(Some(prepend("x")));
    }
    // The prefix cache is keyed by raw input bytes, so it must be emptied.
    // The segment and plain-text caches are keyed by normalized text and
    // could not return stale ids, but would keep unreachable entries.
    assert_eq!(cached.prefix_cache.as_ref().unwrap().len(), 0);
    assert_eq!(segments(&cached), 0);
    for (text, before) in [&chat[..], short].iter().zip(&before) {
        let want = plain.encode(text).unwrap();
        assert_ne!(&want, before, "the normalizer must change the ids");
        assert_eq!(cached.encode(text).unwrap(), want);
    }

    // The plain-text cache, on the scanner path.
    let plain_text = "plain text, the thing\n".repeat(500);
    let mut scan_plain = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    let mut scan_cached = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    scan_cached.enable_input_cache(4);
    let before = scan_cached.encode(&plain_text).unwrap();
    let input_entries = |t: &Tokenizer| {
        t.input_cache
            .as_ref()
            .unwrap()
            .lock()
            .unwrap()
            .entries
            .len()
    };
    assert_eq!(input_entries(&scan_cached), 1);
    for tok in [&mut scan_plain, &mut scan_cached] {
        tok.set_normalizer(Some(prepend("x")));
    }
    assert_eq!(input_entries(&scan_cached), 0);
    let want = scan_plain.encode(&plain_text).unwrap();
    assert_ne!(want, before, "the normalizer must change the ids");
    assert_eq!(scan_cached.encode(&plain_text).unwrap(), want);
}

#[test]
fn prefix_cache_matches_uncached_encoding_on_the_scanner_path() {
    let plain = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    let mut cached = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    cached.enable_input_cache(16);
    let requests = chat_requests(5, &SCAN_SPECIALS, 60);
    for (i, request) in requests.iter().enumerate() {
        assert_eq!(
            cached.encode(request).unwrap(),
            plain.encode(request).unwrap(),
            "request {i}"
        );
    }
}

#[test]
fn prefix_cache_reuses_the_previous_request() {
    let mut tok = prefix_test_tokenizer(Value::Null, prefix_byte_level());
    tok.enable_input_cache(4);
    let cache = tok.prefix_cache.as_ref().unwrap();
    let mut conversation = String::new();
    for turn in 0..6 {
        conversation.push_str(&format!(
            "<|c|>user {turn}: {}<|a|>",
            "the thing ".repeat(500)
        ));
        tok.encode(&conversation).unwrap();
        // One entry per conversation: each request replaces the last.
        assert_eq!(cache.len(), 1, "turn {turn}");
    }
    let chain = PrefixCache::chain(conversation.as_bytes(), 0);
    let hit = cache.lookup(conversation.as_bytes(), &chain).unwrap();
    assert_eq!(hit.lcp, conversation.len());

    // Plain text and short input are not cached here.
    tok.encode(&"the thing ".repeat(1000)).unwrap();
    tok.encode("<|a|>short").unwrap();
    assert_eq!(cache.len(), 1);
}

#[test]
fn prefix_cache_does_not_cut_where_the_new_input_splits_differently() {
    // Bulk byte-level: a text split is one pretoken, so a merge across a
    // wrong cut changes the ids.
    let bulk = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false});
    let plain = prefix_test_tokenizer(Value::Null, bulk.clone());
    let body = "the thing ".repeat(600);
    for (first, second) in [
        // `x<|b|>y` takes the place of `<|b|>` and the `x` before it.
        (
            format!("<|c|>{body}x<|b|>"),
            format!("<|c|>{body}x<|b|>y tail"),
        ),
        // ` <|c|>z` starts in the whitespace `<|c|>` absorbed.
        (
            format!("<|a|>{body}x  <|c|>"),
            format!("<|a|>{body}x  <|c|>z tail"),
        ),
        // `<|a|>>` starts where `<|a|>` does, which is harmless.
        (
            format!("<|c|>{body}<|a|>"),
            format!("<|c|>{body}<|a|>>tail"),
        ),
        // Whitespace the cached input ends with is absorbed by `<|b|>`.
        (
            format!("<|c|>{body}<|a|>  "),
            format!("<|c|>{body}<|a|>  <|b|>tail"),
        ),
    ] {
        let mut cached = prefix_test_tokenizer(Value::Null, bulk.clone());
        cached.enable_input_cache(4);
        assert_eq!(
            cached.encode(&first).unwrap(),
            plain.encode(&first).unwrap()
        );
        assert_eq!(
            cached.encode(&second).unwrap(),
            plain.encode(&second).unwrap(),
            "{second:?}"
        );
    }
}

#[test]
fn prefix_cache_cuts_at_a_token_inside_whitespace_an_rstrip_absorbed() {
    // More than three distinct first bytes: the full-scan split, which
    // matches ` <q>` inside the space `<|b|>`'s `rstrip` absorbed and
    // emits it with a span starting before the end of `<|b|>`'s.
    let mut added = PREFIX_ADDED.to_vec();
    added.extend([(" <q>", false, false, true), ("#t#", false, false, true)]);
    let bulk = json!({"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false});
    let plain = prefix_test_tokenizer_with(&added, Value::Null, bulk.clone());
    let mut cached = prefix_test_tokenizer_with(&added, Value::Null, bulk);
    cached.enable_input_cache(4);
    let body = "the thing ".repeat(600);
    let first = format!("<|c|>{body}a<|b|> <q>");
    let second = format!("{first}tail<|a|>");
    let q = plain.token_to_id(" <q>").unwrap();
    assert!(plain.encode(&first).unwrap().ends_with(&[q]));
    // The two spans really overlap.
    let spans: Vec<TokenSpan> = plain
        .added_tokens()
        .unwrap()
        .split_spans(&first, false)
        .into_iter()
        .filter_map(|seg| match seg {
            SpannedSegment::Token(span) => Some(span),
            SpannedSegment::Text(_) => None,
        })
        .collect();
    let [.., b, last] = spans[..] else {
        panic!("expected at least two added tokens")
    };
    assert_eq!(last.id, q);
    assert!(last.start < b.end, "{b:?} {last:?}");
    for text in [&first, &second] {
        assert_eq!(cached.encode(text).unwrap(), plain.encode(text).unwrap());
    }
}

#[test]
fn prefix_cache_stays_correct_when_tokens_are_added() {
    let mut plain = prefix_test_tokenizer(Value::Null, prefix_byte_level());
    let mut cached = prefix_test_tokenizer(Value::Null, prefix_byte_level());
    cached.enable_input_cache(4);
    let input = format!("<|a|>{}<|d|> now<|a|>", "say hi to ".repeat(500));
    assert_eq!(
        cached.encode(&input).unwrap(),
        plain.encode(&input).unwrap()
    );
    for tok in [&mut plain, &mut cached] {
        tok.add_tokens(&[NewToken::special("<|d|>")]).unwrap();
    }
    assert_eq!(
        cached.encode(&input).unwrap(),
        plain.encode(&input).unwrap()
    );
}
