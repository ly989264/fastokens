#!/usr/bin/env python3
"""Closed-loop replay of many opencode sessions against one tokenizer, for
comparing the encode caches under a large load.

N sessions (see `workload.py`: ~45 KB shared system prompt and tools, tool
results from real files, reasoning dropped at each user turn, auto-compaction
at ~700K chars) each start at a random point of their life; the request
before that point is encoded during warm-up, as a long-running server would
already have done (with --cold every session starts at its first request
instead, as right after a restart). Then requests are taken from a session picked at random
(seeded, so every run sees the same sequence) and encoded one at a time:
`Tokenizer.encode(text).ids`, the call a serving stack makes.

Run once per variant, in its own process, with the cache env vars set:
  python prefix_bench.py --label off --json off.json
  FASTOKENS_SEGMENT_CACHE=2048 python prefix_bench.py --label seg --json seg.json
  FASTOKENS_INPUT_CACHE=512 python prefix_bench.py --label prefix --json prefix.json
  python prefix_bench.py --compare off.json seg.json prefix.json

Each result keeps a checksum of every request's ids; --compare checks they
agree across runs and prints a table.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import resource
import statistics
import sys
import time
import zlib
from array import array

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload  # noqa: E402

MAX_OFFSET = 250  # sessions join at a random request index below this


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


class Session:
    def __init__(self, sid: int, rng: random.Random, compact_at: int, private: bool = False, cold: bool = False):
        self.sid, self.rng, self.generation, self.compact_at = sid, rng, 0, compact_at
        self.private = private
        self._start(offset=0 if cold else rng.randrange(MAX_OFFSET))

    def _view(self, msgs):
        """With `private`, this session's tool results start every line with a
        tag of its own, so no two sessions share file contents."""
        if not self.private:
            return msgs
        tag = f"{self.sid}.{self.generation}| "
        return [dict(m, content=m["content"].replace("\n", "\n" + tag)) if m["role"] == "tool" else m
                for m in msgs]

    def _start(self, offset: int):
        seed = 10_000 * (self.generation + 1) + self.sid
        self.stream = workload.session_requests(seed, compact_at=self.compact_at, stop_after_compactions=3)
        self.prev = None
        for _ in range(offset):
            msgs = next(self.stream, None)
            if msgs is None:
                break
            self.prev = msgs
        if self.prev is not None:
            self.prev = self._view(self.prev)

    def next(self):
        msgs = next(self.stream, None)
        if msgs is None:  # the session ended: a new one starts cold
            self.generation += 1
            self._start(offset=0)
            msgs = next(self.stream)
        msgs = self._view(msgs)
        self.prev, prev = msgs, self.prev
        kind = "new" if prev is None else "compacted" if len(msgs) < len(prev) else "grow"
        if kind == "grow" and msgs[-1]["role"] == "user":
            kind = "user-turn"  # reasoning of earlier turns is dropped here
        return msgs, kind


class ChatSession:
    """One real multi-turn chat (ShareGPT): a request per user turn; when the
    conversation ends, the session moves on to an unused one."""

    def __init__(self, sid: int, rng: random.Random, convs, n_sessions: int, cold: bool = False):
        self.convs, self.ci, self.step = convs, sid, n_sessions
        self.turn = 0 if cold else rng.randrange(1, max(2, len(convs[sid]) // 2))
        self.prev = convs[sid][: 2 * self.turn - 1] if self.turn else None

    def next(self):
        c = self.convs[self.ci % len(self.convs)]
        self.turn += 1
        kind = "chat"
        if 2 * self.turn - 1 > len(c):
            self.ci += self.step
            self.turn, kind = 1, "new"
            c = self.convs[self.ci % len(self.convs)]
        self.prev = c[: 2 * self.turn - 1]
        return self.prev, kind


def run(args):
    tok = workload.load_fastokens(args.model)
    encode = lambda text: tok.encode(text, add_special_tokens=False).ids  # noqa: E731

    rng = random.Random(args.seed)
    if args.chat:
        import prefix_check

        convs = json.load(open(args.chat, encoding="utf-8"))
        repo = workload.MODELS[args.model][0]
        renderer = prefix_check.Renderer(args.model, repo, "template", workload._hub(repo, "tokenizer.json"))
        render = renderer.chat
        sessions = [ChatSession(i, random.Random(rng.random()), convs, args.sessions, args.cold)
                    for i in range(args.sessions)]
    else:
        keep = "all" if args.keep_reasoning else None
        render_fn = workload.MODELS[args.model][1]
        render = (lambda msgs: render_fn(msgs, keep=keep)) if keep else (lambda msgs: render_fn(msgs))  # noqa: E731
        sessions = [Session(i, random.Random(rng.random()), args.compact_at, args.private_files, args.cold)
                    for i in range(args.sessions)]
    if args.plain:
        # The same text without the model's added tokens: plain text, which
        # the chat-template cache only looks up and passes on.
        repo = workload.MODELS[args.model][0]
        added = [t["content"] for t in json.load(open(workload._hub(repo, "tokenizer.json")))["added_tokens"]]
        added.sort(key=len, reverse=True)
        chat_render = render

        def render(msgs):  # noqa: F811
            text = chat_render(msgs)
            for t in added:
                text = text.replace(t, "")
            return text
    if args.unique:
        # Nothing reusable: a per-request nonce starts the input and every line.
        base_render, nonce = render, iter(range(10**9))

        def render(msgs):  # noqa: F811
            n = f"{next(nonce):x}"
            return f"[{n}]\n" + base_render(msgs).replace("\n", f"\n{n} ")
    t = time.perf_counter()
    for s in sessions:
        if s.prev is not None:
            encode(render(s.prev))
    warmup_s = time.perf_counter() - t

    order = random.Random(args.seed + 1)
    records = []  # (ms, tokens, chars, kind)
    checksums = []
    # CPU spent encoding = process CPU (every thread, incl. the encoder's pool
    # spinning down after a request) minus this thread's own work outside the
    # encode call (session bookkeeping, rendering, checksums).
    load_start = os.getloadavg()
    cpu0, other = time.process_time(), 0.0
    for _ in range(args.requests):
        t_other = time.thread_time()
        s = sessions[order.randrange(len(sessions))]
        msgs, kind = s.next()
        text = render(msgs)
        other += time.thread_time() - t_other
        t0 = time.perf_counter()
        ids = encode(text)
        dt = (time.perf_counter() - t0) * 1e3
        t_other = time.thread_time()
        records.append((dt, len(ids), len(text), kind))
        checksums.append(zlib.crc32(array("I", ids).tobytes()))
        del ids
        other += time.thread_time() - t_other
    cpu_s = time.process_time() - cpu0 - other

    total_ms = sum(r[0] for r in records)
    caches = cache_report(tok)
    kinds = sorted({r[3] for r in records})
    return {
        "label": args.label,
        "model": args.model,
        "sessions": args.sessions,
        "requests": len(records),
        "env": {k: v for k, v in os.environ.items() if k.startswith("FASTOKENS_")},
        "warmup_s": round(warmup_s, 2),
        "encode_s": total_ms / 1e3,
        "req_per_s": len(records) / (total_ms / 1e3),
        "mtok_per_s": sum(r[1] for r in records) / (total_ms / 1e3) / 1e6,
        "avg_tokens": sum(r[1] for r in records) / len(records),
        "avg_chars": sum(r[2] for r in records) / len(records),
        "ms_p50": pct([r[0] for r in records], 0.5),
        "ms_p90": pct([r[0] for r in records], 0.9),
        "ms_p99": pct([r[0] for r in records], 0.99),
        "ms_max": max(r[0] for r in records),
        "ms_mean": statistics.fmean(r[0] for r in records),
        "cpu_ms_per_request": cpu_s * 1e3 / len(records),
        "by_kind": {
            k: {
                "n": sum(1 for r in records if r[3] == k),
                "ms_mean": statistics.fmean(r[0] for r in records if r[3] == k),
                "ms_p50": pct([r[0] for r in records if r[3] == k], 0.5),
            }
            for k in kinds
        },
        "load_avg_start_end": [round(load_start[0], 2), round(os.getloadavg()[0], 2)],
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20,
        "rss_mib": psutil.Process().memory_info().rss / 2**20,
        "caches": caches,
        "checksums": checksums,
    }


def cache_report(tok):
    """What the caches report about themselves: sizes in every build, and with
    the `observe` feature also evictions (shard counters summed, histograms
    dropped)."""
    try:
        t = json.loads(tok.observe())
    except AttributeError:  # a build without `observe()`
        return {}
    out = {}
    seg = t.get("segment_cache")
    if seg:
        shards = seg.get("shards") or []
        seg = {k: v for k, v in seg.items() if not isinstance(v, (list, dict))}
        for k in ("rotations", "evicted_entries", "evicted_bytes", "inserts"):
            if shards:
                seg[k] = sum(s[k] for s in shards)
        out["segment_cache"] = seg
    if t.get("prefix_cache"):
        out["prefix_cache"] = t["prefix_cache"]
    return out


def compare(paths):
    runs = [json.load(open(p)) for p in paths]
    base = runs[0]
    print(f"{base['model']}: {base['sessions']} sessions, {base['requests']} requests, "
          f"avg {base['avg_chars'] / 1e3:.0f}K chars / {base['avg_tokens'] / 1e3:.1f}K tokens per request")
    head = (f"{'variant':<14}{'parity':>8}{'encode s':>10}{'req/s':>8}{'Mtok/s':>8}{'p50 ms':>8}"
            f"{'p90 ms':>8}{'p99 ms':>8}{'max ms':>8}{'cpu/req':>9}{'RSS MiB':>9}{'vs ' + base['label']:>10}{'cpu vs':>8}")
    print(head)
    for r in runs:
        same = sum(a == b for a, b in zip(r["checksums"], base["checksums"]))
        parity = "ok" if same == len(base["checksums"]) == len(r["checksums"]) else f"{same}/{len(base['checksums'])}"
        print(f"{r['label']:<14}{parity:>8}{r['encode_s']:>10.1f}{r['req_per_s']:>8.1f}{r['mtok_per_s']:>8.2f}"
              f"{r['ms_p50']:>8.2f}{r['ms_p90']:>8.2f}{r['ms_p99']:>8.2f}{r['ms_max']:>8.1f}"
              f"{r['cpu_ms_per_request']:>9.2f}{r['rss_mib']:>9.0f}{base['encode_s'] / r['encode_s']:>9.2f}x"
              f"{base['cpu_ms_per_request'] / r['cpu_ms_per_request']:>7.2f}x")
    print("\ncache self-reported size at the end (MiB; evictions where the build counts them)")
    for r in runs:
        c = r.get("caches") or {}
        seg, pre = c.get("segment_cache"), (c.get("prefix_cache") or {}).get("chat")
        parts = []
        if seg:
            parts.append(f"segment {seg['bytes'] / 2**20:.0f} MiB / budget {seg['approx_max_bytes'] / 2**20:.0f}, "
                         f"{seg['entries']} entries" + (f", {seg['evicted_entries']} evicted, {seg['rotations']} rotations"
                                                       if "evicted_entries" in seg else ""))
        if pre:
            parts.append(f"prefix {pre['bytes'] / 2**20:.0f} MiB, {pre['entries']}/{pre['capacity']} entries"
                         + (f", {pre['evicted_entries']} evicted, {pre['replaced']} replaced" if "evicted_entries" in pre else ""))
        print(f"  {r['label']:<14}" + ("; ".join(parts) or "-"))
    kinds = sorted(base["by_kind"])
    print("\nmean ms by request kind (n): " + ", ".join(f"{k} ({base['by_kind'][k]['n']})" for k in kinds))
    print(f"{'variant':<14}" + "".join(f"{k:>12}" for k in kinds))
    for r in runs:
        print(f"{r['label']:<14}" + "".join(f"{r['by_kind'][k]['ms_mean']:>12.2f}" for k in kinds))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", default="run")
    ap.add_argument("--model", default="glm-5.2", choices=list(workload.MODELS))
    ap.add_argument("--sessions", type=int, default=256)
    ap.add_argument("--requests", type=int, default=3000, help="measured requests")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compact-at", type=int, default=700_000, help="history size (chars) that triggers compaction")
    ap.add_argument("--unique", action="store_true", help="make every request share nothing with any other")
    ap.add_argument("--keep-reasoning", action="store_true",
                    help="render with every turn's reasoning kept, so no request rewrites history")
    ap.add_argument("--private-files", action="store_true",
                    help="give each session's tool results contents of their own (no sharing across sessions)")
    ap.add_argument("--plain", action="store_true",
                    help="drop the model's added tokens from every request (plain text: what the cache costs where it cannot help)")
    ap.add_argument("--cold", action="store_true",
                    help="start every session at its first request, with nothing encoded before (a fresh server)")
    ap.add_argument("--chat", metavar="JSON", help="replay real chats (ShareGPT sample) with the model's template instead")
    ap.add_argument("--json", help="write the result here")
    ap.add_argument("--compare", nargs="+", metavar="JSON", help="compare saved runs (first is the baseline)")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare)
        return 0
    result = run(args)
    summary = {k: v for k, v in result.items() if k != "checksums"}
    print(json.dumps(summary, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
