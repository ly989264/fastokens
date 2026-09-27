#!/usr/bin/env python3
"""Performance harness for opencode-style agent traffic.

Scenarios (all on chat-templated prompts from `workload.py`):

  single    one request at a given context size; `tmpl` is the real serving
            path (special tokens recognized), `ord` the same bytes through
            encode_ordinary (no special-token split). Every timed call sees a
            distinct prompt, so no whole-input cache can short-circuit it.
  session   one agent session, request after request (each re-sends the full
            history): total / p50 / p99 encode time with no cache, the prefix
            cache (FASTOKENS_INPUT_CACHE) and the segment cache
            (FASTOKENS_SEGMENT_CACHE).
  server    N sessions interleaved round-robin, as a server sees them.
  gil       worst stall of a 1 ms ticker thread (stand-in for the server's
            event loop) while another thread encodes long prompts.
  threads   aggregate throughput with K Python client threads.
  tf        end-to-end through the patched transformers tokenizer.

Usage:
  python examples/agent/bench.py --json base.json            # save a baseline
  python examples/agent/bench.py --json new.json --compare base.json
  python examples/agent/bench.py --quick --models qwen3-coder --scenarios single,session

Timings are medians over repeats/seeds. Build the extension in release mode
(`maturin develop --release`) before running; A/B only numbers from the same
machine, idle, on AC power.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload  # noqa: E402

SCENARIOS = ["single", "session", "server", "gil", "threads", "tf"]
SIZES = [16_000, 64_000, 128_000, 200_000]  # context sizes (tokens) for `single`


def ms(f):
    t = time.perf_counter()
    f()
    return (time.perf_counter() - t) * 1e3


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


# Cache variants for the session/server scenarios: (tag, env vars). The prefix
# cache holds a number of inputs, the segment cache a size in MiB.
def cache_variants(n_inputs):
    return [("nocache", {}), ("cache", {"FASTOKENS_INPUT_CACHE": n_inputs}),
            ("segcache", {"FASTOKENS_SEGMENT_CACHE": 512})]


def make(model, env=None):
    env = env or {}
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        return workload.load_fastokens(model)
    finally:
        for k in env:
            os.environ.pop(k, None)


class Prompts:
    """Per-seed request streams. Message lists are cached (they share their
    message dicts, so they are cheap); rendered prompts are not (a full
    session is tens of MB of text), so callers render just before timing."""

    def __init__(self, model, quick):
        self.model, self.quick, self._s = model, quick, {}

    def requests(self, seed):
        if seed not in self._s:
            kw = {"compact_at": 250_000, "max_requests": 80} if self.quick else {}
            self._s[seed] = list(workload.session_requests(seed, **kw))
        return self._s[seed]

    def render(self, msgs):
        return workload.render(self.model, msgs)

    def stream(self, seed):
        for msgs in self.requests(seed):
            yield self.render(msgs)

    def last(self, seed, n=1, step=1):
        return [self.render(m) for m in self.requests(seed)[-n::step]]


def warm(tok, prompts):
    """Populate word-level caches on inputs unrelated to what gets timed."""
    for p in prompts:
        tok.encode(p)
        tok.encode_ordinary(p)


def bench_single(model, P, args, out):
    tok = make(model)
    seeds = range(100, 100 + 2 * args.repeat)  # single requests are the noisiest metric
    warm(tok, P.last(99, 3))
    # per seed: the first request whose token count reaches each target size
    picks = {size: [] for size in SIZES}
    for seed in seeds:
        todo = list(SIZES)
        for p in P.stream(seed):
            n = len(tok.encode(p).ids)
            while todo and n >= todo[0]:
                picks[todo.pop(0)].append((p, n))
            if not todo:
                break
    for size in SIZES:
        if not picks[size]:
            continue
        n = statistics.median(c for _, c in picks[size])
        t = statistics.median(ms(lambda: tok.encode(p)) for p, _ in picks[size])
        o = statistics.median(ms(lambda: tok.encode_ordinary(p)) for p, _ in picks[size])
        k = f"{size // 1000}k"
        out[f"single.tmpl_ms@{k}"] = t
        out[f"single.ord_ms@{k}"] = o
        print(f"    single  ~{int(n):7d} tok   tmpl {t:7.2f} ms ({n / t / 1e3:5.1f} Mtok/s)   ord {o:7.2f} ms ({n / o / 1e3:5.1f} Mtok/s)")


def bench_session(model, P, args, out):
    for tag, env in cache_variants(64):
        totals, p50s, p99s, lasts = [], [], [], []
        for seed in range(200, 200 + args.repeat):
            tok = make(model, env)
            warm(tok, P.last(99, 2))
            lat = [ms(lambda: tok.encode(p)) for p in P.stream(seed)]
            totals.append(sum(lat))
            p50s.append(pct(lat, .5))
            p99s.append(pct(lat, .99))
            lasts.append(max(lat))
        out[f"session.total_ms.{tag}"] = statistics.median(totals)
        out[f"session.p50_ms.{tag}"] = statistics.median(p50s)
        out[f"session.p99_ms.{tag}"] = statistics.median(p99s)
        n = len(P.requests(200))
        print(f"    session {tag:8} {n} requests: total {statistics.median(totals):7.0f} ms   "
              f"p50 {statistics.median(p50s):6.2f}   p99 {statistics.median(p99s):6.2f}   max {statistics.median(lasts):6.2f} ms")


def bench_server(model, P, args, out):
    n_sess = 4 if args.quick else 16
    streams = [P.requests(300 + s) for s in range(n_sess)]
    order = [(s, i) for i in range(max(map(len, streams))) for s in range(n_sess) if i < len(streams[s])]
    for tag, env in cache_variants(2 * n_sess):
        tok = make(model, env)
        warm(tok, P.last(99, 2))
        lat = []
        for s, i in order:
            p = P.render(streams[s][i])
            lat.append(ms(lambda: tok.encode(p)))
        out[f"server{n_sess}.total_ms.{tag}"] = sum(lat)
        out[f"server{n_sess}.p99_ms.{tag}"] = pct(lat, .99)
        print(f"    server  {n_sess} sessions {tag:8} {len(order)} requests: total {sum(lat):7.0f} ms   "
              f"p50 {pct(lat, .5):6.2f}   p99 {pct(lat, .99):6.2f} ms")


def bench_gil(model, P, args, out):
    tok = make(model)
    big = [P.last(400 + s)[0] for s in range(args.repeat)]
    warm(tok, P.last(99, 2))
    for name, fn in [("encode", lambda p: tok.encode(p)),
                     ("encode_batch_flat", lambda p: tok.encode_batch_flat([p]))]:
        stop = threading.Event()

        def loop():
            i = 0
            while not stop.is_set():
                fn(big[i % len(big)])
                i += 1
        th = threading.Thread(target=loop)
        th.start()
        gaps, last, end = [], time.perf_counter(), time.perf_counter() + (1.0 if args.quick else 3.0)
        while time.perf_counter() < end:
            time.sleep(0.001)
            now = time.perf_counter()
            gaps.append((now - last) * 1e3)
            last = now
        stop.set()
        th.join()
        out[f"gil.{name}.max_gap_ms"] = max(gaps)
        out[f"gil.{name}.p99_gap_ms"] = pct(gaps, .99)
        print(f"    gil     {name:18} 1 ms ticker: worst gap {max(gaps):7.2f} ms   p99 {pct(gaps, .99):6.2f}   median {pct(gaps, .5):5.2f} ms")


def bench_threads(model, P, args, out):
    tok = make(model)
    pool = [p for s in range(500, 500 + (4 if args.quick else 8)) for p in P.last(s, 40, 10)]
    ntok = sum(len(tok.encode(p).ids) for p in pool)
    for k in (1, 2, 4, 8):
        best = float("inf")
        for _ in range(2):
            it = iter(range(len(pool)))
            lock = threading.Lock()

            def work():
                while True:
                    with lock:
                        i = next(it, None)
                    if i is None:
                        return
                    tok.encode(pool[i])
            ths = [threading.Thread(target=work) for _ in range(k)]
            t = time.perf_counter()
            for th in ths:
                th.start()
            for th in ths:
                th.join()
            best = min(best, time.perf_counter() - t)
        out[f"threads.mtoks@{k}"] = ntok / best / 1e6
    print("    threads encode, " + "   ".join(f"K={k}: {out[f'threads.mtoks@{k}']:5.1f}" for k in (1, 2, 4, 8))
          + f" Mtok/s  ({len(pool)} prompts, {ntok // len(pool)} tok avg)")


def bench_tf(model, P, args, out):
    if workload.MODELS[model][2] != "hf":
        return
    import fastokens
    from transformers import PreTrainedTokenizerFast

    fastokens.patch_transformers()
    try:
        tf = PreTrainedTokenizerFast(tokenizer_file=workload._hub(workload.MODELS[model][0], "tokenizer.json"))
    finally:
        fastokens.unpatch_transformers()
    tok = make(model)
    prompts = [P.last(600 + s)[0] for s in range(args.repeat)]
    warm(tok, P.last(99, 2))
    a = statistics.median(ms(lambda: tok.encode(p).ids) for p in prompts)
    b = statistics.median(ms(lambda: tf(p, add_special_tokens=False)["input_ids"]) for p in prompts)
    out["tf.native_ms"], out["tf.transformers_ms"] = a, b
    print(f"    tf      native encode {a:6.2f} ms   via transformers {b:6.2f} ms   (+{b - a:.2f} ms)")


def meta():
    def git(*a):
        try:
            return subprocess.run(["git", "-C", workload.REPO, *a], capture_output=True, text=True).stdout.strip()
        except OSError:
            return ""
    import fastokens

    return {"git": git("rev-parse", "--short", "HEAD") + ("-dirty" if git("status", "--porcelain", "--", "src", "python") else ""),
            "fastokens": os.path.dirname(fastokens.__file__), "machine": platform.machine(),
            "cpu": (subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
                    if sys.platform == "darwin" else platform.processor()),
            "cpus": os.cpu_count(), "bpe_threads": os.environ.get("FASTOKENS_BPE_THREADS", "default"),
            "corpus": workload.CORPUS_COMMIT[:12], "time": time.strftime("%Y-%m-%d %H:%M:%S")}


# metrics where larger is better; everything else is a time
HIGHER_IS_BETTER = ("threads.mtoks",)
# Re-running unchanged code moves aggregate metrics (session/server totals,
# p50/p99, throughput, >=64k single requests) by <5%, sub-ms single requests by
# up to ~15%, and a single worst-case sample (max_gap) arbitrarily, so: flag
# only changes beyond NOISE, and leave max_gap out of the comparison.
NOISE = 0.10
UNCOMPARED = ("max_gap_ms",)
# Ticker gaps sit on a ~1.5 ms floor (1 ms sleep + wakeup), where fractions of
# a millisecond are scheduler jitter: flag them only beyond this many ms.
GAP_NOISE_MS = 1.0


def compare(new, base_path):
    with open(base_path) as f:
        base = json.load(f)
    print(f"\n== compare vs {base_path} ({base['meta'].get('git')} -> {new['meta'].get('git')})")
    print(f"  {'model':14} {'metric':34} {'base':>10} {'new':>10} {'change':>8}")
    for model, metrics in new["results"].items():
        for k, v in metrics.items():
            b = base["results"].get(model, {}).get(k)
            if b is None or b == 0 or k.endswith(UNCOMPARED):
                continue
            speedup = (v / b) if k.startswith(HIGHER_IS_BETTER) else (b / v)
            flag = "  faster" if speedup > 1 + NOISE else "  SLOWER" if speedup < 1 - NOISE else ""
            if k.endswith("_gap_ms") and abs(v - b) < GAP_NOISE_MS:
                flag = ""
            print(f"  {model:14} {k:34} {b:10.2f} {v:10.2f} {speedup:7.2f}x{flag}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(workload.OPENCODE_MODELS), help="comma-separated, from: " + ", ".join(workload.MODELS))
    ap.add_argument("--scenarios", default=",".join(SCENARIOS), help="comma-separated, from: " + ", ".join(SCENARIOS))
    ap.add_argument("--repeat", type=int, default=5, help="seeds/repeats per measurement (default 5)")
    ap.add_argument("--quick", action="store_true", help="shorter sessions and fewer repeats")
    ap.add_argument("--json", help="write results to this file")
    ap.add_argument("--compare", help="baseline JSON to compare against")
    args = ap.parse_args()
    if args.quick:
        args.repeat = min(args.repeat, 3)

    m = meta()
    print(" ".join(f"{k}={v}" for k, v in m.items()))
    results = {}
    for model in args.models.split(","):
        print(f"  {model}")
        P = Prompts(model, args.quick)
        out = results[model] = {}
        for sc in args.scenarios.split(","):
            globals()[f"bench_{sc}"](model, P, args, out)
    doc = {"meta": m, "results": results}
    if args.json:
        with open(args.json, "w") as f:
            json.dump(doc, f, indent=1)
        print(f"wrote {args.json}")
    if args.compare:
        compare(doc, args.compare)
    return 0


if __name__ == "__main__":
    sys.exit(main())
