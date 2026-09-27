#!/usr/bin/env python3
"""Replay realistic opencode traffic against a serving-style frontend.

Models what a vLLM-like OpenAI server does with coding-agent traffic:

* N concurrent agent sessions (see `workload.py`). Each is a loop: send the
  whole history, wait for the model and the tool (log-normal, median 3 s),
  send again. After the assistant hands back to the user, the user thinks
  (median 20 s). Histories grow until auto-compaction replaces them with a
  summary; a session that ends is replaced by a new one (cold start).
* Steady state: each session starts at a random point of its life, and the
  request before that point is encoded during warm-up, as a long-running
  server would already have done.
* The server is one asyncio event loop that renders each request (chat
  template) and hands tokenization to a thread pool, calling the tokenizer
  through `transformers`, as serving stacks do. A 5 ms ticker on the loop
  stands in for the SSE streaming every other session is waiting on.

Reported per run: requests and tokens served in the window, the latency
tokenization adds to each request (queueing + encode), the event-loop lag,
and the process CPU time spent per request.

Usage (the tokenizer implementation is whatever `fastokens` / `tokenizers`
the running interpreter imports):
  python examples/agent/simulate.py --impl hf --sessions 32 --json hf-32.json
  python examples/agent/simulate.py --impl fastokens --sessions 32 --json ft-32.json
  FASTOKENS_SEGMENT_CACHE=1024 python examples/agent/simulate.py --impl fastokens ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import resource
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload  # noqa: E402

STEP_MEDIAN_S = 3.0  # model generation + tool execution per agent step
THINK_MEDIAN_S = 20.0  # user reading the answer and typing the next turn
MAX_OFFSET = 250  # steady state: sessions join at a random request index below this


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else float("nan")


class Session:
    """One opencode session's request stream, replaced by a fresh session
    (new seed, cold start) when it ends."""

    def __init__(self, sid: int, rng: random.Random):
        self.sid, self.rng, self.generation = sid, rng, 0
        self._start(offset=rng.randrange(MAX_OFFSET))

    def _start(self, offset: int):
        seed = 10_000 * (self.generation + 1) + self.sid
        self.stream = workload.session_requests(seed, stop_after_compactions=3)
        self.prev = None
        for _ in range(offset):  # fast-forward to the join point
            msgs = next(self.stream, None)
            if msgs is None:
                break
            self.prev = msgs

    def next(self):
        msgs = next(self.stream, None)
        if msgs is None:
            self.generation += 1
            self._start(offset=0)
            msgs = next(self.stream)
        self.prev, prev = msgs, self.prev
        return msgs, prev


def build_tokenizer(model: str, impl: str):
    from transformers import PreTrainedTokenizerFast

    path = workload._hub(workload.MODELS[model][0], "tokenizer.json")
    if impl == "fastokens":
        import fastokens

        fastokens.patch_transformers()
    return PreTrainedTokenizerFast(tokenizer_file=path)


async def run(args):
    rng = random.Random(args.seed)
    model = args.model
    tok = build_tokenizer(model, args.impl)
    encode = lambda text: tok(text, add_special_tokens=False)["input_ids"]  # noqa: E731

    sessions = [Session(i, random.Random(rng.random())) for i in range(args.sessions)]
    # Warm-up: the request each session sent before joining, so caches hold
    # what a long-running server would hold. Not measured.
    t = time.perf_counter()
    for s in sessions:
        if s.prev is not None:
            encode(workload.render(model, s.prev))
    warmup_s = time.perf_counter() - t

    pool = ThreadPoolExecutor(max_workers=args.workers)
    loop = asyncio.get_running_loop()
    stop_at = loop.time() + args.duration
    records = []  # (queue_ms, encode_ms, tokens, kind)
    lags = []
    lock = threading.Lock()

    def job(text, submitted):
        started = time.perf_counter()
        n = len(encode(text))
        done = time.perf_counter()
        return (started - submitted) * 1e3, (done - started) * 1e3, n

    async def agent(s: Session):
        await asyncio.sleep(s.rng.uniform(0, STEP_MEDIAN_S))
        while loop.time() < stop_at:
            msgs, prev = s.next()
            kind = ("new" if prev is None else "compacted" if len(msgs) < len(prev) else "grow")
            text = workload.render(model, msgs)
            queue_ms, encode_ms, n = await loop.run_in_executor(pool, job, text, time.perf_counter())
            if loop.time() <= stop_at:
                with lock:
                    records.append((queue_ms, encode_ms, n, kind))
            user_turn = msgs[-1]["role"] == "user" and prev is not None
            median = THINK_MEDIAN_S if user_turn else STEP_MEDIAN_S
            await asyncio.sleep(median * s.rng.lognormvariate(0, 0.5))

    async def ticker():
        period = 0.005
        expected = loop.time() + period
        while loop.time() < stop_at:
            await asyncio.sleep(max(0.0, expected - loop.time()))
            lags.append((loop.time() - expected) * 1e3)
            expected += period

    cpu0 = resource.getrusage(resource.RUSAGE_SELF)
    wall0 = time.perf_counter()
    tasks = [asyncio.create_task(agent(s)) for s in sessions] + [asyncio.create_task(ticker())]
    # Stop at the deadline: requests still queued then count as backlog.
    await asyncio.sleep(args.duration)
    pending_at_stop = sum(1 for t in tasks if not t.done())
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    wall = time.perf_counter() - wall0
    cpu1 = resource.getrusage(resource.RUSAGE_SELF)
    pool.shutdown(wait=True, cancel_futures=True)

    total = [q + e for q, e, _, _ in records]
    tokens = sum(n for _, _, n, _ in records)
    cpu_s = (cpu1.ru_utime - cpu0.ru_utime) + (cpu1.ru_stime - cpu0.ru_stime)
    by_kind = {k: sum(1 for r in records if r[3] == k) for k in ("grow", "compacted", "new")}
    return {
        "impl": args.label or args.impl, "model": model, "sessions": args.sessions,
        "workers": args.workers, "duration_s": round(wall, 1), "warmup_s": round(warmup_s, 1),
        "requests": len(records), "requests_by_kind": by_kind,
        "req_per_s": len(records) / wall, "mtok_per_s": tokens / wall / 1e6,
        "avg_tokens": tokens / max(1, len(records)),
        "added_ms_p50": pct(total, .5), "added_ms_p90": pct(total, .9),
        "added_ms_p99": pct(total, .99), "added_ms_max": max(total, default=float("nan")),
        "encode_ms_p50": pct([e for _, e, _, _ in records], .5),
        "encode_ms_p99": pct([e for _, e, _, _ in records], .99),
        "queue_ms_p99": pct([q for q, _, _, _ in records], .99),
        "loop_lag_ms_p50": pct(lags, .5), "loop_lag_ms_p99": pct(lags, .99),
        "loop_lag_ms_max": max(lags, default=float("nan")),
        "cpu_ms_per_request": cpu_s * 1e3 / max(1, len(records)),
        "cpu_cores_busy": cpu_s / wall, "sessions_waiting_at_stop": pending_at_stop - 1,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", choices=["hf", "fastokens"], required=True)
    ap.add_argument("--label", help="name for this run in the output (default: --impl)")
    ap.add_argument("--model", default="glm-5.2", choices=list(workload.MODELS))
    ap.add_argument("--sessions", type=int, default=32)
    ap.add_argument("--duration", type=float, default=90.0, help="measured seconds")
    ap.add_argument("--workers", type=int, default=4, help="tokenizer thread-pool size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="write the result here")
    args = ap.parse_args()
    result = asyncio.run(run(args))
    print(json.dumps(result, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
