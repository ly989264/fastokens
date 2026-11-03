#!/usr/bin/env python3
"""Concurrency soak for the prefix cache: many Python threads share cached
tokenizers whose capacity is far below the working set, so lookups, inserts,
replacements and evictions race constantly. Requests are drawn in random order
from a pool of agent-session requests (so a thread may go back to an older,
shorter history, or repeat one exactly), mixed with `encode_batch`,
`split_special_tokens` and `encode_ordinary`. Every result is compared with an
uncached tokenizer; RSS is sampled to check that memory stays flat.

  python prefix_soak.py --model glm-5.2 --threads 16 --duration 300
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import threading
import time

import psutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload  # noqa: E402


def build(model, env):
    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        return workload.load_fastokens(model)
    finally:
        for k in env:
            os.environ.pop(k, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="glm-5.2")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--duration", type=float, default=300)
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--per-session", type=int, default=40)
    args = ap.parse_args()

    pool = []  # per session: list of rendered requests, in order
    for seed in range(args.sessions):
        reqs = []
        for msgs in workload.session_requests(100 + seed, compact_at=150_000):
            reqs.append(workload.render(args.model, msgs))
            if len(reqs) == args.per_session:
                break
        pool.append(reqs)
    plain = build(args.model, {})
    shared = {
        "cap4": build(args.model, {"FASTOKENS_INPUT_CACHE": 4}),
        "cap4+seg2M": build(args.model, {"FASTOKENS_INPUT_CACHE": 4, "FASTOKENS_SEGMENT_CACHE": 2}),
        "cap64": build(args.model, {"FASTOKENS_INPUT_CACHE": 64}),
    }
    names = list(shared)
    stop = time.monotonic() + args.duration
    ops, errors, lock = [0], [], threading.Lock()

    def worker(k):
        rng = random.Random(k)
        cursor = [rng.randrange(len(s)) for s in pool]
        n = 0
        while time.monotonic() < stop:
            s = rng.randrange(len(pool))
            r = rng.random()
            if r < 0.7:  # the session's next request
                cursor[s] = (cursor[s] + 1) % len(pool[s])
            elif r < 0.85:  # back to an older, shorter history
                cursor[s] = rng.randrange(len(pool[s]))
            text = pool[s][cursor[s]]  # else: exact repeat
            name = names[rng.randrange(len(names))]
            tok = shared[name]
            op = rng.random()
            if op < 0.75:
                got, want = tok.encode(text).ids, plain.encode(text).ids
            elif op < 0.85:
                got = tok.encode(text, split_special_tokens=True).ids
                want = plain.encode(text, split_special_tokens=True).ids
            elif op < 0.9:
                got, want = tok.encode_ordinary(text).ids, plain.encode_ordinary(text).ids
            else:
                batch = [pool[b][rng.randrange(len(pool[b]))] for b in rng.sample(range(len(pool)), 4)]
                got = [e.ids for e in tok.encode_batch(batch)]
                want = [e.ids for e in plain.encode_batch(batch)]
            if got != want:
                with lock:
                    errors.append(f"{name} session {s} request {cursor[s]} op {op:.2f}")
            n += 1
        with lock:
            ops[0] += n

    proc = psutil.Process()
    rss = []
    ths = [threading.Thread(target=worker, args=(k,)) for k in range(args.threads)]
    t0 = time.monotonic()
    for th in ths:
        th.start()
    while any(th.is_alive() for th in ths):
        rss.append(proc.memory_info().rss / 2**20)
        time.sleep(5)
    for th in ths:
        th.join()
    pool_mb = sum(len(t) for s in pool for t in s) / 1e6
    half = len(rss) // 2
    print(f"{args.model}: {args.threads} threads, {time.monotonic() - t0:.0f}s, {ops[0]} operations, "
          f"{len(errors)} wrong; pool {pool_mb:.0f} MB of text; RSS MiB first half max "
          f"{max(rss[:half] or [0]):.0f}, second half max {max(rss[half:] or [0]):.0f}, final {rss[-1]:.0f}")
    for e in errors[:5]:
        print("   ", e)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
