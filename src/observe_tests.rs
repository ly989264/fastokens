//! `observe` counters as the encode pipeline reports them: the path each input
//! takes, per-thread encode counts, the segment cache's counters and the pool
//! jobs a long input or a batch starts.

use serde_json::{Value, json};

use crate::{
    observe,
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
    assert_eq!(a["prefix_cache"], false);
    let c = |v: &Value, k: &str| n(v, &["segment_cache", k]);
    // Text segments of CHAT: "user\nHello there", "assistant\nthe thing".
    assert_eq!((c(&a, "hits"), c(&a, "misses")), (0, 2));
    assert_eq!((c(&b, "hits"), c(&b, "misses")), (2, 2));
    assert_eq!(c(&b, "hit_bytes"), c(&a, "miss_bytes"));
    assert_eq!(c(&b, "entries"), 2);
    assert!(c(&b, "miss_tokenize_ns") > 0);
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
