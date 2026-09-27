#!/usr/bin/env python3
"""Correctness harness for opencode-style agent traffic.

Replays interleaved agent sessions (see `workload.py`) through every fastokens
entry point a server uses and requires bit-identical token ids against the
reference implementation (HF `tokenizers`, or `tiktoken` for Kimi):

  encode           chat-templated prompt, special tokens recognized
  encode+cache     same, with the prefix cache on, fed the interleaved stream
                   in order (so cache state carries across requests/sessions)
  split_special    split_special_tokens=True (untrusted-text path)
  ordinary+cache   encode_ordinary with the cache on == without it
  batch            encode_batch == reference
  threads          several Python threads sharing one cached tokenizer
  decode           decode(ids) == reference decode, with/without skipping specials
  transformers     (--transformers) patched transformers tokenizer(prompt)

Usage:
  python examples/agent/check.py                 # all models, 3 interleaved sessions
  python examples/agent/check.py --quick         # 1 short session per model
  python examples/agent/check.py --models qwen3-coder,gpt-oss --transformers

Exit status is non-zero if any check diverges.
"""

from __future__ import annotations

import argparse
import os
from array import array
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload  # noqa: E402

CHUNK = 16  # prompts per reference encode_batch call


def interleaved_stream(n_sessions: int, quick: bool):
    """[(session, request_index, messages)] round-robin across sessions."""
    kw = {"compact_at": 150_000, "max_requests": 60} if quick else {}
    gens = [workload.session_requests(seed, **kw) for seed in range(n_sessions)]
    out, idx, live = [], [0] * n_sessions, list(range(n_sessions))
    while live:
        for s in list(live):
            msgs = next(gens[s], None)
            if msgs is None:
                live.remove(s)
                continue
            out.append((s, idx[s], msgs))
            idx[s] += 1
    return out


def with_cache(model: str, capacity: int):
    os.environ["FASTOKENS_INPUT_CACHE"] = str(capacity)
    try:
        return workload.load_fastokens(model)
    finally:
        del os.environ["FASTOKENS_INPUT_CACHE"]


class Check:
    def __init__(self, name: str):
        self.name, self.n, self.tokens, self.fails, self.first, self.secs = name, 0, 0, 0, None, 0.0

    def compare(self, where: str, got, exp, ref: workload.Reference):
        """`exp` is an array('I') (full-size token lists as Python ints would
        take GBs); `got` may be a list or an array."""
        self.n += 1
        self.tokens += len(exp)
        if not isinstance(got, array):
            got = array("I", got)
        if got == exp:
            return
        self.fails += 1
        got, exp = got.tolist(), exp.tolist()
        if self.first is None:
            i = next((k for k, (a, b) in enumerate(zip(got, exp)) if a != b), min(len(got), len(exp)))
            ctx = ref.decode(exp[max(0, i - 6): i + 6], False)
            self.first = (f"{where} token {i} (len fast {len(got)} vs ref {len(exp)}): "
                          f"fast {got[i:i + 4]} ref {exp[i:i + 4]} ctx {ctx!r}")

    def line(self) -> str:
        status = "ok" if self.fails == 0 else f"FAIL {self.fails}/{self.n}"
        s = f"    {self.name:15} {status:>12}   {self.n:5d} cases {self.tokens / 1e6:8.2f}M tok  {self.secs:6.1f}s"
        return s + (f"\n        first mismatch: {self.first}" if self.first else "")


def check_model(model: str, args) -> bool:
    t0 = time.perf_counter()
    stream = interleaved_stream(args.sessions, args.quick)
    prompts = [workload.render(model, msgs) for _, _, msgs in stream]
    where = [f"session {s} request {r}" for s, r, _ in stream]
    ref = workload.Reference(model)
    fast = workload.load_fastokens(model)
    exp = []
    for i in range(0, len(prompts), CHUNK):
        exp += [array("I", ids) for ids in ref.encode_batch(prompts[i: i + CHUNK])]
    print(f"  {model}: {len(prompts)} requests from {args.sessions} interleaved sessions, "
          f"{sum(map(len, exp)) / 1e6:.1f}M reference tokens ({time.perf_counter() - t0:.1f}s to build)")

    checks = []

    def run(name, body):
        c = Check(name)
        t = time.perf_counter()
        body(c)
        c.secs = time.perf_counter() - t
        checks.append(c)
        print(c.line(), flush=True)

    run("encode", lambda c: [c.compare(w, fast.encode(p).ids, e, ref) for w, p, e in zip(where, prompts, exp)])

    cached = with_cache(model, 2 * args.sessions)
    run("encode+cache", lambda c: [c.compare(w, cached.encode(p).ids, e, ref) for w, p, e in zip(where, prompts, exp)])

    def split_special(c):
        sel = list(range(0, len(prompts), 8))
        refs = ref.encode_text_batch([prompts[i] for i in sel])
        for i, e in zip(sel, refs):
            c.compare(where[i], fast.encode(prompts[i], split_special_tokens=True).ids, array("I", e), ref)
    run("split_special", split_special)

    cached_ord = with_cache(model, 2 * args.sessions)
    run("ordinary+cache", lambda c: [c.compare(w, cached_ord.encode_ordinary(p).ids, array("I", fast.encode_ordinary(p).ids), ref)
                                     for w, p in zip(where, prompts)])

    def batch(c):
        for i in range(0, len(prompts), 8):
            for j, enc in enumerate(fast.encode_batch(prompts[i: i + 8])):
                c.compare(where[i + j], enc.ids, exp[i + j], ref)
    run("batch", batch)

    def threads(c):
        shared = with_cache(model, 2 * args.sessions)
        results = [None] * len(prompts)

        def worker(k):
            for i in range(k, len(prompts), args.threads):
                results[i] = array("I", shared.encode(prompts[i]).ids)
        ths = [threading.Thread(target=worker, args=(k,)) for k in range(args.threads)]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        for w, got, e in zip(where, results, exp):
            c.compare(w, got, e, ref)
    run("threads", threads)

    def decode(c):
        last = {s: i for i, (s, _, _) in enumerate(stream)}
        for i in last.values():
            ids = exp[i].tolist()
            for skip in (False, True):
                got, want = fast.decode(ids, skip_special_tokens=skip), ref.decode(ids, skip)
                c.n += 1
                c.tokens += len(exp[i])
                if got != want:
                    c.fails += 1
                    k = next((k for k, (a, b) in enumerate(zip(got, want)) if a != b), min(len(got), len(want)))
                    c.first = c.first or f"{where[i]} skip={skip} char {k}: fast {got[k:k + 30]!r} ref {want[k:k + 30]!r}"
    run("decode", decode)

    if args.transformers and workload.MODELS[model][2] == "hf":
        import fastokens
        from transformers import PreTrainedTokenizerFast

        fastokens.patch_transformers()
        try:
            tf = PreTrainedTokenizerFast(tokenizer_file=workload._hub(workload.MODELS[model][0], "tokenizer.json"))
        finally:
            fastokens.unpatch_transformers()
        run("transformers", lambda c: [c.compare(where[i], tf(prompts[i], add_special_tokens=False)["input_ids"], exp[i], ref)
                                       for i in range(0, len(prompts), 16)])

    return all(c.fails == 0 for c in checks)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(workload.MODELS), help="comma-separated, from: " + ", ".join(workload.MODELS))
    ap.add_argument("--sessions", type=int, default=3, help="interleaved sessions per model (default 3)")
    ap.add_argument("--threads", type=int, default=4, help="threads for the concurrency check (default 4)")
    ap.add_argument("--quick", action="store_true", help="1 short session per model")
    ap.add_argument("--transformers", action="store_true", help="also check the patched transformers path")
    args = ap.parse_args()
    if args.quick:
        args.sessions = 1

    import fastokens

    print(f"fastokens from {os.path.dirname(fastokens.__file__)}; corpus @ {workload.CORPUS_COMMIT[:12]}")
    ok = True
    for model in args.models.split(","):
        try:
            ok &= check_model(model, args)
        except Exception as e:  # noqa: BLE001 - report and continue with the next model
            print(f"  {model}: ERROR {type(e).__name__}: {e}")
            ok = False
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
