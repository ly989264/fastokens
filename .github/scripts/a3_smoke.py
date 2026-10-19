#!/usr/bin/env python3
"""Smoke test of an installed fastokens wheel, on the machine it runs on (the
A3 workflow runs it on arm64 Linux for each variant). Needs no network: the
tokenizers are built here.

  python a3_smoke.py --variant default|observe [--perf]

1. `fastokens.observe.enabled()` matches the variant.
2. Two tokenizers (a byte-level BPE with a regex Split, the fused path; and an
   o200k-pattern tiktoken model, the scanner path), each with special tokens,
   encode multi-turn chat requests (each one the previous plus a turn) with
   the segment cache, the prefix cache and both turned on (and both with tiny
   budgets); every encode and encode_batch must give the ids of a tokenizer
   without caches, each cache must hold entries afterwards (except under the
   tiny budgets) and the prefix cache must stay within its byte budget.
3. observe variant: `observe.layers()` on an encode, and a snapshot whose
   cache counters saw the traffic.
4. --perf: `python -m fastokens.observe sample` on an encode loop (through
   `perf` on Linux). If no sampler is installed or it may not run, prints a
   warning and exits 0.
"""

import argparse
import base64
import json
import os
import random
import subprocess
import sys
import tempfile

SPECIALS = ["<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>"]
CACHE_ENV = ("FASTOKENS_SEGMENT_CACHE", "FASTOKENS_INPUT_CACHE", "FASTOKENS_INPUT_CACHE_MIB")
GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def byte_level_json() -> str:
    """A small byte-level BPE (GPT-2 style Split + ByteLevel) with merges for
    common English and the specials as added tokens."""
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: i for i, b in enumerate(range(256))}
    merges = []
    words = ["the", " the", "ing", " in", "and", " and", " to", "er", " fix", " bug", " read", " file",
             " def", " return", " self", " tool", " call", "ion", " test", " code"]
    for w in words:
        pieces = [b2u[b] for b in w.encode()]
        while len(pieces) > 1:
            a, b = pieces[0], pieces[1]
            if a + b not in vocab:
                vocab[a + b] = len(vocab)
                merges.append(f"{a} {b}")
            pieces = [a + b] + pieces[2:]
    added = [{"id": len(vocab) + i, "content": s, "single_word": False, "lstrip": False, "rstrip": False,
              "normalized": False, "special": True} for i, s in enumerate(SPECIALS)]
    return json.dumps({
        "version": "1.0", "truncation": None, "padding": None, "added_tokens": added, "normalizer": None,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": GPT2_PATTERN}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": None,
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": False,
                  "vocab": vocab, "merges": merges},
    })


def tiktoken_model(path: str) -> None:
    ranks = [bytes([i]) for i in range(256)] + [b"th", b"the", b" the", b"in", b"ing", b" th", b"he", b"ll",
                                                b"hell", b"hello", b" t", b"er", b" f", b" fi", b" fix"]
    with open(path, "w") as f:
        f.write("".join(f"{base64.b64encode(t).decode()} {i}\n" for i, t in enumerate(ranks)))


def requests(seed: int, turns: int = 24) -> list[str]:
    """Chat-template-like requests of one session: each request is the
    previous conversation plus a new turn, ending with an open assistant turn."""
    rng = random.Random(seed)
    words = ["read", "src/lib.rs", "fix", "the", "bug", "def", "encode(self):", "return", "ids", "tool", "call",
             "testing", "and", "file", "123", "4567", "ünïcödé", "日本語", "\n", "    ", "\t"]
    conv, out = "<|im_start|>system\nYou are a coding agent.<|im_end|>\n", []
    for t in range(turns):
        role = ["user", "assistant", "tool"][t % 3]
        body = " ".join(rng.choice(words) for _ in range(rng.randint(150, 400)))
        if role == "assistant":
            body = f"<think>{body[:200]}</think>{body}<tool_call>read_file</tool_call>"
        conv += f"<|im_start|>{role}\n{body}<|im_end|>\n"
        out.append(conv + "<|im_start|>assistant\n")
    return out


def make(build, env: dict):
    """build() with only the cache env vars in `env` set (they are read when a
    tokenizer is created)."""
    saved = {k: os.environ.pop(k) for k in CACHE_ENV if k in os.environ}
    os.environ.update(env)
    try:
        return build()
    finally:
        for k in CACHE_ENV:
            os.environ.pop(k, None)
        os.environ.update(saved)


def check_caches(variant: str) -> list:
    from fastokens import _native
    tmp = tempfile.mkdtemp()
    model = os.path.join(tmp, "o200k-mini.tiktoken")
    tiktoken_model(model)
    builders = {
        "byte-level BPE (fused split)": lambda: _native.Tokenizer.from_json_str(byte_level_json()),
        "tiktoken o200k (scanner)": lambda: _native.Tokenizer.from_tiktoken(
            model, encoding="o200k_base", special_tokens={s: 400 + i for i, s in enumerate(SPECIALS)}),
    }
    configs = {
        "segment cache": {"FASTOKENS_SEGMENT_CACHE": "4"},
        "prefix cache": {"FASTOKENS_INPUT_CACHE": "8"},
        "both": {"FASTOKENS_SEGMENT_CACHE": "4", "FASTOKENS_INPUT_CACHE": "8"},
        "both, tiny budgets": {"FASTOKENS_SEGMENT_CACHE": "1", "FASTOKENS_INPUT_CACHE": "2",
                               "FASTOKENS_INPUT_CACHE_MIB": "1"},
    }
    sessions = [requests(s) for s in range(3)]
    flat = [r for k in range(max(map(len, sessions))) for s in sessions for r in s[k:k + 1]]   # sessions interleaved
    kept = []
    for tname, build in builders.items():
        ref_tok = make(build, {})
        ref = [ref_tok.encode(p).ids for p in flat]
        assert any(len(r) > 1000 for r in ref)
        for cname, env in configs.items():
            tok = make(build, env)
            for rnd in range(2):                     # cold, then everything seen before
                for p, r in zip(flat, ref):
                    assert tok.encode(p).ids == r, f"{tname} / {cname}: encode differs"
            batch = [e.ids for e in tok.encode_batch(flat[:12])]
            assert batch == ref[:12], f"{tname} / {cname}: encode_batch differs"
            ob = json.loads(tok.observe())
            seg, pre = ob.get("segment_cache"), ob.get("prefix_cache")
            tiny = "FASTOKENS_INPUT_CACHE_MIB" in env    # entries may all be too big to keep; ids must still match
            if "FASTOKENS_SEGMENT_CACHE" in env:
                assert seg and seg["entries"] > 0, f"{tname} / {cname}: segment cache unused: {seg}"
            if "FASTOKENS_INPUT_CACHE" in env:
                chat = (pre or {}).get("chat") or {}
                assert tiny or chat.get("entries", 0) > 0, f"{tname} / {cname}: prefix cache unused: {pre}"
                assert chat["bytes"] <= chat["max_bytes"], f"{tname} / {cname}: prefix cache over its budget: {chat}"
            print(f"ok  {tname:30} {cname:18} {len(flat)} requests x2 + encode_batch, identical ids; "
                  f"segment cache {seg and seg['entries']} entries, prefix cache "
                  f"{pre and pre['chat']['entries']} entries / {pre and pre['chat']['bytes']} bytes")
            kept.append(tok)
    return kept


def check_observe(toks) -> None:
    import fastokens.observe as obs
    tok = toks[-1]
    text = requests(9)[-1]
    r = obs.layers(lambda: tok.encode(text).ids, calls=20)
    assert r["layers_ms"]["rust"] > 0 and r["layers_ms"]["lists"] > 0, r["layers_ms"]
    print(obs.layers_report(r, "smoke"))
    snap = obs.snapshot({"t": tok})
    assert snap["process"]["enabled"], snap["process"]
    sc = snap["tokenizers"]["t"]["segment_cache"]
    assert sc["hits"] + sc["misses"] > 0, sc
    print(f"ok  observe snapshot: segment cache hits {sc['hits']} misses {sc['misses']}, "
          f"threads {len(snap['process']['threads'])}")


def check_perf() -> None:
    from fastokens import _profile
    tool = _profile.sampler_available()
    if tool is None:
        print("::warning::observe sample: no CPU sampler here (Linux needs `perf`); skipped")
        return
    loop = ("import fastokens._native as n\n"
            f"t = n.Tokenizer.from_json_str({byte_level_json()!r})\n"
            f"text = {requests(7)[-1]!r}\n"
            "while True:\n"
            "    t.encode(text).ids\n")
    out = os.path.join(tempfile.mkdtemp(), "sample-smoke")
    r = subprocess.run([sys.executable, "-m", "fastokens.observe", "sample", "--seconds", "3", "--delay", "1",
                        "--out", out, "--", sys.executable, "-c", loop], capture_output=True, text=True, timeout=300)
    if r.returncode != 0:
        msg = (r.stderr.strip().splitlines() or ["no output"])[-1]
        print(r.stderr[-3000:])
        print(f"::warning::observe sample through {tool} did not run: {msg}")
        return
    with open(out + ".json") as f:
        summ = json.load(f)
    print(f"ok  observe sample through {tool}: {summ['samples']} samples, {summ['on_cpu_samples']} on CPU")
    with open(out + ".txt") as f:
        print("\n".join(f.read().splitlines()[:45]))
    if summ["on_cpu_samples"] == 0:
        print(f"::warning::observe sample through {tool} recorded no on-CPU samples")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["default", "observe"], required=True)
    ap.add_argument("--perf", action="store_true")
    a = ap.parse_args()
    import fastokens
    import fastokens.observe as obs
    print(f"fastokens from {os.path.dirname(fastokens.__file__)}, observe enabled: {obs.enabled()}")
    assert obs.enabled() == (a.variant == "observe"), f"variant {a.variant} but observe.enabled() is {obs.enabled()}"
    if a.perf:
        check_perf()
        return
    toks = check_caches(a.variant)
    if a.variant == "observe":
        check_observe(toks)
    print("smoke test passed")


if __name__ == "__main__":
    main()
