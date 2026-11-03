//! `observe` counters as the encode pipeline reports them: the path each input
//! takes, per-thread encode counts, the segment cache's counters and the pool
//! jobs a long input or a batch starts.

use serde_json::{Value, json};

use crate::{
    observe,
    prefix_cache_tests::{prefix_byte_level, prefix_test_tokenizer},
    segment_cache_tests::{scan_test_tokenizer, segment_test_tokenizer},
    tiktoken,
};

/// Runs `f` on a new thread, whose observe slot is its own: exact per-thread
/// counts hold while other tests run.
fn on_fresh_thread<R: Send + 'static>(f: impl FnOnce() -> R + Send + 'static) -> R {
    std::thread::spawn(f).join().unwrap()
}

fn n(v: &Value, path: &[&str]) -> u64 {
    path.iter()
        .fold(v, |v, k| &v[*k])
        .as_u64()
        .unwrap_or_else(|| panic!("{path:?} missing in {v}"))
}

const CHAT: &str =
    "<|im_start|>user\nHello there<|im_end|><|im_start|>assistant\nthe thing<|im_end|>";

#[test]
fn encode_paths_and_bytes_are_counted_exactly_per_thread() {
    on_fresh_thread(|| {
        let tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        let plain = "Hello the world\n".repeat(20);
        tok.encode(&plain).unwrap();
        tok.encode(CHAT).unwrap();
        tok.encode(CHAT).unwrap();
        let mut cached = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        cached.enable_segment_cache(1 << 20);
        cached.encode(CHAT).unwrap();
        cached.encode(CHAT).unwrap();
        // One plain segment: the cache does not apply.
        cached.encode(&plain).unwrap();
        let general = segment_test_tokenizer(
            json!({"type": "NFC"}),
            json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}),
        );
        general.encode("the thing").unwrap();
        // Empty input returns before the pipeline: not counted.
        tok.encode("").unwrap();

        let me = observe::current_thread();
        assert_eq!(n(&me, &["paths", "scan_whole"]), 2, "{me}");
        assert_eq!(n(&me, &["paths", "scan_splits"]), 2, "{me}");
        assert_eq!(n(&me, &["paths", "segments_cached"]), 2, "{me}");
        assert_eq!(
            n(&me, &["paths", "general"]) + n(&me, &["paths", "fused_split"]),
            1,
            "{me}"
        );
        assert_eq!(n(&me, &["encodes"]), 7);
        assert_eq!(
            n(&me, &["encode_bytes"]),
            (2 * plain.len() + 4 * CHAT.len() + "the thing".len()) as u64
        );
        assert!(n(&me, &["encode_ns"]) > 0);
    });
}

#[test]
fn a_repeated_request_is_all_hits() {
    let mut tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    tok.enable_segment_cache(1 << 20);
    let first = tok.encode(CHAT).unwrap();
    let a = tok.observe_json();
    let second = tok.encode(CHAT).unwrap();
    let b = tok.observe_json();
    assert_eq!(first, second);
    assert_eq!(a["enabled"], true);
    assert_eq!(a["scanner_path"], true);
    assert!(a["prefix_cache"].is_null());
    let c = |v: &Value, k: &str| n(v, &["segment_cache", k]);
    // Text segments of CHAT: "user\nHello there", "assistant\nthe thing".
    assert_eq!((c(&a, "hits"), c(&a, "misses")), (0, 2));
    assert_eq!((c(&b, "hits"), c(&b, "misses")), (2, 2));
    assert_eq!(c(&b, "hit_bytes"), c(&a, "miss_bytes"));
    assert_eq!(c(&b, "entries"), 2);
    assert!(c(&b, "miss_tokenize_ns") > 0);
    // All hits: no miss tokenizing is timed.
    assert_eq!(c(&b, "miss_tokenize_ns"), c(&a, "miss_tokenize_ns"));
    let plain = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    assert!(plain.observe_json()["segment_cache"].is_null());
}

#[test]
fn long_inputs_and_batches_are_counted_as_pool_jobs() {
    let tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    let line = "Hello the world, this is the thing\n";
    let plain = line.repeat(256 * 1024 / line.len());
    let chat = format!("<|im_start|>{plain}<|im_end|><|im_start|>{plain}<|im_end|>");
    let before = observe::snapshot();
    let one = tok.encode(&plain).unwrap();
    let two = tok.encode(&chat).unwrap();
    let batch = tok
        .encode_batch(&[chat.as_str(), "a", "the"], false)
        .unwrap();
    let after = observe::snapshot();
    assert_eq!(batch[0], two);
    assert!(!one.is_empty());
    let d = |kind: &str, k: &str| n(&after, &["jobs", kind, k]) - n(&before, &["jobs", kind, k]);
    assert!(d("batch", "jobs") >= 1);
    assert!(d("batch", "tasks") >= 3);
    if crate::fanout::threads() >= 2 {
        // The plain input, and each long text segment of the chat input (in
        // `encode` and again in the batch), is scanned in parallel chunks.
        assert!(d("scan", "jobs") >= 5, "{after}");
        let workers: Vec<&Value> = after["threads"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| t["name"].as_str().unwrap().starts_with("fastokens-"))
            .collect();
        assert!(!workers.is_empty());
        assert!(workers.iter().all(|t| t["kind"] == "worker"));
        assert!(workers.iter().map(|t| n(t, &["tasks"])).sum::<u64>() > 0);
    }
}

/// `me`'s counter at `path` minus `before`'s.
fn d(me: &Value, before: &Value, path: &[&str]) -> u64 {
    n(me, path) - n(before, path)
}

const STAGES_FOR_TEST: [&str; 8] = [
    "added_tokens",
    "normalize",
    "prefix_cache",
    "pretokenize",
    "model",
    "cache_lookup",
    "cache_assemble",
    "post_process",
];

#[test]
fn stages_land_on_the_path_that_ran_them() {
    on_fresh_thread(|| {
        let tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        let a = observe::current_thread();
        tok.encode(CHAT).unwrap();
        let b = observe::current_thread();
        let st = |k: &str| d(&b, &a, &["stage_ns", k]);
        assert!(st("added_tokens") > 0 && st("model") > 0, "{b}");
        assert_eq!(
            st("cache_lookup") + st("cache_assemble") + st("prefix_cache"),
            0
        );
        let total: u64 = STAGES_FOR_TEST.iter().map(|k| st(k)).sum();
        assert!(total <= d(&b, &a, &["encode_ns"]), "{total} > encode");

        let mut cached = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        cached.enable_segment_cache(1 << 20);
        cached.encode(CHAT).unwrap();
        let c = observe::current_thread();
        assert!(d(&c, &b, &["stage_ns", "cache_lookup"]) > 0);
        assert!(
            d(&c, &b, &["stage_ns", "model"]) > 0,
            "misses are tokenized"
        );
        cached.encode(CHAT).unwrap();
        let e = observe::current_thread();
        assert!(d(&e, &c, &["stage_ns", "cache_lookup"]) > 0);
        assert_eq!(
            d(&e, &c, &["stage_ns", "model"]),
            0,
            "all hits: no model work"
        );
        assert_eq!(d(&e, &c, &["paths", "segments_cached"]), 1);
        // Nested stages are part of the outer one: the sum stays within the
        // encode time on every path.
        let total: u64 = STAGES_FOR_TEST
            .iter()
            .map(|k| d(&e, &b, &["stage_ns", k]))
            .sum();
        assert!(total <= d(&e, &b, &["encode_ns"]), "{total} > encode");

        let normalized = segment_test_tokenizer(
            json!({"type": "NFC"}),
            json!({"type": "ByteLevel", "add_prefix_space": true, "trim_offsets": true, "use_regex": true}),
        );
        let text = "the thing in the ring ".repeat(500);
        let f = observe::current_thread();
        normalized.encode(&text).unwrap();
        let g = observe::current_thread();
        assert!(d(&g, &f, &["stage_ns", "normalize"]) > 0);
        assert!(d(&g, &f, &["stage_ns", "model"]) > 0);
    });
}

#[test]
fn bpe_cache_counters_follow_the_work() {
    on_fresh_thread(|| {
        let tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        // Under 64 KiB, so it all runs on this thread.
        let text = (0..400).map(|i| format!("word{i} ")).collect::<String>();
        let a = observe::current_thread();
        let first = tok.encode(&text).unwrap();
        let b = observe::current_thread();
        let second = tok.encode(&text).unwrap();
        let c = observe::current_thread();
        assert_eq!(first, second);
        let first_lookups = d(&b, &a, &["bpe", "fused_lookups"]);
        assert!(first_lookups >= 400, "{b}");
        assert!(d(&b, &a, &["bpe", "fused_misses"]) > 0);
        assert!(d(&b, &a, &["bpe", "fused_misses"]) <= first_lookups);
        assert!(d(&b, &a, &["bpe", "fused_merge_ns"]) > 0);
        // The same pretokens again, all in this thread's cache now.
        assert_eq!(d(&c, &b, &["bpe", "fused_lookups"]), first_lookups);
        assert_eq!(d(&c, &b, &["bpe", "fused_misses"]), 0);
        assert_eq!(d(&c, &b, &["bpe", "fused_merge_ns"]), 0);

        // A pretoken over 15 bytes has no inline key; the long-pretoken cache
        // keeps it, so only the first encode merges it.
        let long = "a".repeat(40);
        tok.encode(&long).unwrap();
        tok.encode(&long).unwrap();
        let e = observe::current_thread();
        assert_eq!(d(&e, &c, &["bpe", "fused_misses"]), 1);
        assert_eq!(d(&e, &c, &["bpe", "fused_uncacheable"]), 0);
        assert_eq!(d(&e, &c, &["bpe", "fused_miss_bytes"]), 40);
        // One led by a CJK character is kept by no cache: merged every time.
        let cjk = "中文模型测试分词缓存"; // 30 bytes, one pretoken
        tok.encode(cjk).unwrap();
        tok.encode(cjk).unwrap();
        let f = observe::current_thread();
        assert_eq!(d(&f, &e, &["bpe", "fused_misses"]), 2);
        assert_eq!(d(&f, &e, &["bpe", "fused_uncacheable"]), 2);
        assert_eq!(d(&f, &e, &["bpe", "fused_miss_bytes"]), 60);
    });
}

#[test]
fn prefix_cache_counts_repeats_reuse_and_misses() {
    on_fresh_thread(|| {
        let mut tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        tok.enable_input_cache(4);
        let plain = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        let base: String = (0..1500)
            .map(|i| format!("line {i}: the thing\n"))
            .collect();
        let longer = format!("{base}Hello the end\n");
        for text in [&base, &base, &longer] {
            assert_eq!(tok.encode(text).unwrap(), plain.encode(text).unwrap());
        }
        let p = &tok.observe_json()["prefix_cache"];
        assert_eq!(n(p, &["lookups"]), 3);
        assert_eq!(n(p, &["misses"]), 1);
        assert_eq!(n(p, &["exact_hits"]), 1);
        assert_eq!(n(p, &["prefix_hits"]), 1);
        assert!(n(p, &["reused_tokens"]) >= 256);
        // The exact repeat reused all of `base`, the prefix hit nearly all.
        assert!(n(p, &["reused_bytes"]) > base.len() as u64);
        assert_eq!(
            n(p, &["lookup_bytes"]),
            (2 * base.len() + longer.len()) as u64
        );
        // Only the miss stores an encoding.
        assert_eq!(n(p, &["entries"]), 1);
        let me = observe::current_thread();
        assert!(n(&me, &["stage_ns", "prefix_cache"]) > 0);
        assert!(plain.observe_json()["prefix_cache"].is_null());
    });
}

#[test]
fn prefix_cache_counts_chat_template_repeats_reuse_and_misses() {
    on_fresh_thread(|| {
        let mut tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        tok.enable_input_cache(4);
        let plain = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        let base: String = (0..300)
            .map(|i| format!("<|im_start|>user\nline {i}: the thing<|im_end|>\n"))
            .collect();
        let longer = format!("{base}<|im_start|>assistant\nHello<|im_end|>\n");
        for text in [&base, &base, &longer] {
            assert_eq!(tok.encode(text).unwrap(), plain.encode(text).unwrap());
        }
        let p = &tok.observe_json()["prefix_cache"];
        assert_eq!(n(p, &["lookups"]), 3);
        assert_eq!(n(p, &["misses"]), 1);
        assert_eq!(n(p, &["exact_hits"]), 1);
        assert_eq!(n(p, &["prefix_hits"]), 1);
        assert!(n(p, &["reused_bytes"]) > base.len() as u64);
        assert_eq!(
            n(p, &["lookup_bytes"]),
            (2 * base.len() + longer.len()) as u64
        );
        // The miss stored `base`, which `longer` then replaced.
        assert_eq!(n(p, &["entries"]), 1);
        let me = observe::current_thread();
        assert!(n(&me, &["stage_ns", "prefix_cache"]) > 0);
        // The exact repeat encoded nothing: counted on its own path.
        assert_eq!(n(&me, &["paths", "prefix_exact"]), 1, "{me}");

        // Too short for the chat path, and without added tokens.
        tok.encode("<|im_start|>hi<|im_end|>").unwrap();
        tok.encode(&"plain text line\n".repeat(400)).unwrap();
        let p = &tok.observe_json()["prefix_cache"];
        let c = &p["chat"];
        // Neither is counted as a lookup here.
        assert_eq!((n(c, &["lookups"]), n(c, &["misses"])), (3, 1));
        assert_eq!((n(c, &["bypass_short"]), n(c, &["bypass_plain"])), (1, 1));
        assert_eq!(
            (
                n(c, &["inserts"]),
                n(c, &["replaced"]),
                n(c, &["evicted_entries"])
            ),
            (2, 1, 0)
        );
        assert_eq!(
            n(c, &["reused_bytes"]) + n(c, &["encoded_bytes"]),
            n(c, &["lookup_bytes"])
        );
        // The totals add the plain-text cache's lookup of the plain text.
        assert_eq!(n(p, &["lookups"]), 4);

        // A normalizer change drops the entry, counted as cleared.
        tok.set_normalizer(None);
        let c = &tok.observe_json()["prefix_cache"]["chat"];
        assert_eq!((n(c, &["entries"]), n(c, &["cleared_entries"])), (0, 1));
        assert_eq!(
            n(c, &["inserts"]),
            n(c, &["replaced"]) + n(c, &["evicted_entries"]) + n(c, &["cleared_entries"])
        );
    });
}

#[test]
fn prefix_cache_counts_cut_backoffs_and_evictions() {
    on_fresh_thread(|| {
        let plain = prefix_test_tokenizer(Value::Null, prefix_byte_level());
        let mut tok = prefix_test_tokenizer(Value::Null, prefix_byte_level());
        tok.enable_input_cache(2);
        let body = |w: &str| format!("{w} ").repeat(1500);
        let inputs = [
            // `<|b|>` ends the cached input; the next one turns it
            // into part of `x<|b|>y`, so reuse backs off to `<|a|>`.
            format!("<|c|>{}<|a|>x<|b|>", body("one")),
            format!("<|c|>{}<|a|>x<|b|>y tail", body("one")),
            // The only shared token is ruled out the same way.
            format!("{}x<|b|>", body("two")),
            format!("{}x<|b|>y tail", body("two")),
            // A third conversation evicts the oldest of two entries.
            format!("<|a|>{}<|a|>", body("three")),
        ];
        for text in &inputs {
            assert_eq!(tok.encode(text).unwrap(), plain.encode(text).unwrap());
        }
        let c = &tok.observe_json()["prefix_cache"]["chat"];
        assert_eq!(n(c, &["cut_backoffs"]), 1);
        assert_eq!(n(c, &["no_cut"]), 1);
        assert_eq!((n(c, &["prefix_hits"]), n(c, &["misses"])), (1, 4));
        assert_eq!(n(c, &["inserts"]), 5);
        assert_eq!(n(c, &["evicted_entries"]), 5 - n(c, &["replaced"]) - 2);
        assert!(n(c, &["evicted_bytes"]) > 0);
        assert_eq!(n(c, &["entries"]), 2);
    });
}

#[test]
fn copies_sharing_a_segment_cache_report_its_id() {
    let mut tok = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    tok.enable_segment_cache(1 << 20);
    let mut copy = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
    copy.share_segment_cache_from(&tok);
    let other = {
        let mut t = scan_test_tokenizer(tiktoken::O200K_BASE_PATTERN);
        t.enable_segment_cache(1 << 20);
        t
    };
    let id = |t: &crate::Tokenizer| t.observe_json()["segment_cache"]["id"].clone();
    assert!(id(&tok).is_u64());
    assert_eq!(id(&tok), id(&copy));
    assert_ne!(id(&tok), id(&other));
    // `add_tokens` moves the copy to a fresh cache of its own.
    copy.add_tokens(&[crate::NewToken::special("<|extra|>")])
        .unwrap();
    assert_ne!(id(&tok), id(&copy));
}
