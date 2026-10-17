//! Differential fuzz of the prefix cache (`enable_input_cache`) on a real
//! tokenizer: random multi-session request streams built from the model's own
//! added tokens, fragments of them and assorted text. Every request is encoded
//! by several cache configurations and compared with an uncached tokenizer.
//!
//! Ignored by default: it needs a `tokenizer.json` and runs for a while.
//!
//! ```sh
//! FUZZ_TOKENIZER=path/to/tokenizer.json FUZZ_SECS=60 FUZZ_SEED=1 \
//!     cargo test --release --test prefix_cache_fuzz -- --ignored --nocapture
//! ```
//!
//! `FUZZ_STRIP=1` first gives the added tokens `lstrip`/`rstrip` flags and adds
//! tokens that contain, extend or overlap the chat tokens, which real chat
//! models rarely have but the cache has to get right.
use std::time::Instant;

use fastokens::{AddedTokenPolicy, Tokenizer};
use serde_json::{Value, json};

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }

    fn chance(&mut self, p: f64) -> bool {
        (self.next() % 1_000_000) as f64 / 1e6 < p
    }
}

fn char_cuts(s: &str) -> Vec<usize> {
    (1..s.len()).filter(|&i| s.is_char_boundary(i)).collect()
}

fn floor_boundary(s: &str, mut i: usize) -> usize {
    while !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

fn env<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// Contents of the chat-looking added tokens of `added`.
fn chat_tokens<'a>(added: impl Iterator<Item = &'a str>) -> Vec<String> {
    const KEYS: [&str; 20] = [
        "user",
        "assistant",
        "system",
        "im_",
        "think",
        "tool",
        "turn",
        "inst",
        "start",
        "end",
        "observation",
        "sop",
        "gmask",
        "message",
        "channel",
        "arg",
        "bos",
        "eos",
        "<s>",
        "</s>",
    ];
    added
        .filter(|c| {
            let s = c.to_lowercase();
            KEYS.iter().any(|k| s.contains(k))
        })
        .map(str::to_string)
        .collect()
}

/// `FUZZ_STRIP`: `lstrip`/`rstrip` on a pattern of the added tokens, and new
/// tokens around the chat ones (`x<tok>y`, `<tok>>`, ` <tok>z`, half a token).
/// Whitespace-led tokens get no `lstrip`, which would hit a known panic of the
/// added-token split outside the cache.
fn with_strip_tokens(mut json: Value) -> Value {
    let added = json["added_tokens"].as_array_mut().expect("added_tokens");
    for (i, t) in added.iter_mut().enumerate() {
        let ws = t["content"]
            .as_str()
            .unwrap()
            .starts_with(char::is_whitespace);
        t["lstrip"] = json!(!ws && (i % 3 == 0 || i % 7 == 0));
        t["rstrip"] = json!(i % 3 == 1 || i % 7 == 0);
    }
    let mut next_id = added
        .iter()
        .map(|t| t["id"].as_u64().unwrap())
        .max()
        .unwrap_or(0)
        + 1;
    let chat = chat_tokens(added.iter().map(|t| t["content"].as_str().unwrap()));
    let mut extra = Vec::new();
    for c in chat.iter().take(12) {
        let half: String = c.chars().take(c.chars().count() / 2).collect();
        extra.extend([format!("x{c}y"), format!("{c}>"), format!(" {c}z"), half]);
    }
    for (j, c) in extra.into_iter().enumerate() {
        if c.is_empty() || added.iter().any(|t| t["content"] == c.as_str()) {
            continue;
        }
        let ws = c.starts_with(char::is_whitespace);
        added.push(json!({"id": next_id, "content": c, "single_word": false,
                          "lstrip": !ws && j % 2 == 0, "rstrip": j % 3 == 0,
                          "normalized": false, "special": j % 2 == 1}));
        next_id += 1;
    }
    json
}

#[test]
#[ignore = "needs FUZZ_TOKENIZER=<tokenizer.json>; runs for FUZZ_SECS"]
fn prefix_cache_matches_uncached_encoding_on_random_sessions() {
    let Ok(path) = std::env::var("FUZZ_TOKENIZER") else {
        eprintln!("FUZZ_TOKENIZER is not set; nothing to fuzz");
        return;
    };
    let secs: f64 = env("FUZZ_SECS", 60.0);
    let seed: u64 = env("FUZZ_SEED", 1);
    let strip = std::env::var("FUZZ_STRIP").is_ok_and(|v| v != "0");
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

    let mut json: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
    if strip {
        json = with_strip_tokens(json);
    }
    let build = || Tokenizer::from_json(json.clone()).unwrap();
    let plain = build();
    let mut variants: Vec<(&str, Tokenizer)> = Vec::new();
    // (name, capacity, prefix-cache budget, segment-cache budget)
    for (name, cap, budget, seg) in [
        ("cap64", 64, None, 0),
        ("cap1", 1, None, 0),
        ("cap64+seg", 64, None, 64 << 20),
        ("cap3+seg64K", 3, None, 64 << 10),
        // Turns away inputs costing over 150 KB and evicts by bytes.
        ("cap64+budget", 64, Some(64 * 150_000), 0),
    ] {
        let mut t = build();
        match budget {
            Some(max_bytes) => t.enable_input_cache_with_budget(cap, max_bytes),
            None => t.enable_input_cache(cap),
        }
        if seg > 0 {
            t.enable_segment_cache(seg);
        }
        variants.push((name, t));
    }

    // Pieces: added tokens (chat-looking ones always, plus a sample), their
    // fragments, glued pairs, and text.
    let configs = plain.added_token_configs().to_vec();
    let mut tokens: Vec<String> = chat_tokens(configs.iter().map(|c| c.content.as_str()));
    tokens.truncate(120);
    for _ in 0..80 {
        if configs.is_empty() {
            break;
        }
        tokens.push(configs[rng.below(configs.len())].content.clone());
    }
    tokens.sort();
    tokens.dedup();
    let strip_tokens = configs.iter().filter(|c| c.lstrip || c.rstrip).count();
    let mut pieces: Vec<String> = Vec::new();
    for t in &tokens {
        pieces.push(t.clone());
        pieces.push(t.clone());
        let cuts = char_cuts(t);
        if !cuts.is_empty() {
            let c = cuts[rng.below(cuts.len())];
            pieces.push(t[..c].to_string());
            pieces.push(t[c..].to_string());
        }
    }
    for _ in 0..tokens.len() {
        let a = &tokens[rng.below(tokens.len())];
        let b = &tokens[rng.below(tokens.len())];
        pieces.push(format!("{a}{b}"));
        pieces.push(format!("x{a}"));
        pieces.push(format!("{a} "));
        pieces.push(format!(" {a}"));
    }
    for t in [
        " ",
        "  ",
        "\n",
        "\n\n",
        "\t",
        "\r\n",
        "\u{3000}",
        " \n",
        "the quick brown fox ",
        "def f(x):\n    return x\n",
        "中文测试，",
        "é",
        "😀",
        "123",
        "4567",
        "'s",
        " don't",
        "{\"a\": 1}",
        "</",
        "<",
        ">",
        "|",
        "<|",
        "|>",
        "▁",
        "Ġ",
        "x",
        "\u{a0}",
        "  \n  ",
        "fn main() {}\n",
        "SELECT * FROM t;",
        "~~~",
        "...",
        "<<",
        ">>",
    ] {
        pieces.push(t.to_string());
    }

    // Sessions share a common "system prompt" so the cache sees cross-session
    // prefixes; each continues, truncates or rewrites its previous request.
    let mut system = String::new();
    while system.len() < 6000 {
        system.push_str(&pieces[rng.below(pieces.len())]);
    }
    let n_sessions = 6;
    let mut last: Vec<String> = vec![String::new(); n_sessions];
    let start = Instant::now();
    let (mut requests, mut bytes, mut shared, mut panics) = (0u64, 0u64, 0u64, 0u64);
    let default_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if std::env::var("FUZZ_VERBOSE_PANIC").is_ok() {
            default_hook(info)
        }
    }));
    while start.elapsed().as_secs_f64() < secs {
        let s = rng.below(n_sessions);
        let prev = &last[s];
        let mut text = if prev.is_empty() || rng.chance(0.03) {
            system.clone()
        } else if rng.chance(0.1) {
            // Rewrite: keep a random prefix, drop the rest (history rewrite).
            let cut = floor_boundary(prev, rng.below(prev.len()));
            prev[..cut].to_string()
        } else if rng.chance(0.6) {
            // Cut near the end (often inside or right after a token).
            let back = rng.below(prev.len().min(96));
            prev[..floor_boundary(prev, prev.len() - back)].to_string()
        } else {
            prev.clone()
        };
        let lcp_ref = text.len().min(prev.len());
        if !rng.chance(0.05) {
            // Mostly small additions; sometimes a big one.
            let target = text.len()
                + if rng.chance(0.1) {
                    20_000 + rng.below(60_000)
                } else {
                    rng.below(2_000)
                };
            while text.len() < target {
                text.push_str(&pieces[rng.below(pieces.len())]);
            }
        }
        if text.len() > 400_000 {
            text = system.clone();
        }
        shared += lcp_ref as u64;
        let policy = if rng.chance(0.2) {
            AddedTokenPolicy::SkipSpecial
        } else {
            AddedTokenPolicy::All
        };
        let add_special = rng.chance(0.3);
        // A panic in the uncached tokenizer is a bug outside the cache; then
        // every variant must panic too.
        let run = |t: &Tokenizer| {
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                t.encode_with_policy(&text, add_special, policy).unwrap()
            }))
        };
        let Ok(want) = run(&plain) else {
            for (name, v) in &variants {
                assert!(
                    run(v).is_err(),
                    "{name} did not panic where uncached encode did"
                );
            }
            panics += 1;
            requests += 1;
            last[s] = String::new(); // continuations would repeat the same panic
            continue;
        };
        for (name, v) in &variants {
            let got = run(v).unwrap();
            if got != want {
                let dump = std::env::temp_dir().join(format!("prefix-cache-fuzz-{seed}.txt"));
                let _ = std::fs::write(&dump, format!("{prev}\n=====PREV/TEXT=====\n{text}"));
                let i = got
                    .iter()
                    .zip(&want)
                    .position(|(a, b)| a != b)
                    .unwrap_or(got.len().min(want.len()));
                panic!(
                    "{path} variant {name} request {requests} policy {policy:?} add_special \
                     {add_special}: first diff at id {i} (got {:?} want {:?}); previous request \
                     and this one written to {}",
                    &got[i..(i + 5).min(got.len())],
                    &want[i..(i + 5).min(want.len())],
                    dump.display()
                );
            }
        }
        if requests % 50 == 49 {
            // Concurrent batch through the shared caches.
            let batch: Vec<&str> = last
                .iter()
                .filter(|t| !t.is_empty())
                .map(String::as_str)
                .collect();
            let want = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                batch
                    .iter()
                    .map(|t| plain.encode(t).unwrap())
                    .collect::<Vec<_>>()
            }));
            if let Ok(want) = want {
                for (name, v) in &variants {
                    assert_eq!(
                        v.encode_batch(&batch, false).unwrap(),
                        want,
                        "batch mismatch in {name}"
                    );
                }
            }
        }
        bytes += text.len() as u64;
        requests += 1;
        last[s] = text;
    }
    println!(
        "OK {requests} requests ({panics} panicked identically with and without the cache), \
         {:.1} MB, avg {:.0} KB/request, {:.0}% bytes shared with the session's previous request, \
         {} added tokens ({strip_tokens} lstrip/rstrip), {} pieces, {:.0}s",
        bytes as f64 / 1e6,
        bytes as f64 / requests.max(1) as f64 / 1e3,
        100.0 * shared as f64 / bytes.max(1) as f64,
        configs.len(),
        pieces.len(),
        start.elapsed().as_secs_f64()
    );
}
