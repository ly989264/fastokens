#!/usr/bin/env python3
"""Drive opencode-style traffic through one shared tokenizer from several
Python threads and serve fastokens' observe dashboard while it runs.

Needs fastokens built with the `observe` feature
(`maturin build --release --features observe`).

  python observe_load.py --threads 8 --cache-mib 256        # then open http://127.0.0.1:9464
  python observe_load.py --threads 4 --cache-mib 0 --api batch
  python -m fastokens.observe top                            # text view from another terminal

--api: encode (one request per call), batch (encode_batch of --batch requests),
mixed (both, plus decode of the result). --gap-ms sleeps between a thread's
requests. Each thread replays its own sessions in order, starting new ones as
they end, like a server. Prompts are rendered in Python on the same threads,
which holds the GIL, so the dashboard's "GIL wait" shows that cost too.
"""
import argparse, itertools, os, random, sys, threading, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import workload
from fastokens._native import Tokenizer
import fastokens.observe as obs

MODELS = {"glm-5.2": ("zai-org/GLM-5.2", workload._glm5)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--sessions", type=int, default=4, help="concurrent sessions per thread")
    ap.add_argument("--cache-mib", type=int, default=256, help="segment cache budget, 0 = off")
    ap.add_argument("--api", choices=["encode", "batch", "mixed"], default="encode")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--gap-ms", type=float, default=0.0)
    ap.add_argument("--duration", type=float, default=0, help="seconds, 0 = until Ctrl-C")
    ap.add_argument("--port", type=int, default=9464)
    ap.add_argument("--model", choices=sorted(MODELS), default="glm-5.2")
    ap.add_argument("--report-every", type=float, default=0, help="also print a text report every N s")
    a = ap.parse_args()
    if not obs.enabled():
        print("warning: this fastokens build has no `observe` feature; the dashboard will be empty", file=sys.stderr)

    repo, render = MODELS[a.model]
    # The segment cache is configured when a tokenizer is created.
    if a.cache_mib:
        os.environ["FASTOKENS_SEGMENT_CACHE"] = str(a.cache_mib)
    else:
        os.environ.pop("FASTOKENS_SEGMENT_CACHE", None)
    tok = Tokenizer.from_file(workload._hub(repo, "tokenizer.json"))
    server = obs.serve({a.model: tok}, port=a.port)
    print(f"dashboard: http://127.0.0.1:{a.port}   json: /json   prometheus: /metrics", flush=True)

    stop = threading.Event()
    seeds = itertools.count(100_000)
    lock = threading.Lock()
    done = [0]

    def next_seed():
        with lock:
            return next(seeds)

    def worker(k):
        rng = random.Random(k)
        sessions = [workload.session_requests(next_seed()) for _ in range(a.sessions)]
        while not stop.is_set():
            prompts = []
            for _ in range(a.batch if a.api != "encode" else 1):
                i = rng.randrange(len(sessions))
                msgs = next(sessions[i], None)
                if msgs is None:
                    sessions[i] = workload.session_requests(next_seed())
                    msgs = next(sessions[i])
                prompts.append(render(msgs))
            if a.api == "encode":
                tok.encode(prompts[0])
            elif a.api == "batch":
                tok.encode_batch(prompts)
            else:
                ids = tok.encode(prompts[0]).ids
                tok.encode_batch(prompts[1:])
                tok.decode(ids[-2000:])
            with lock:
                done[0] += len(prompts)
            if a.gap_ms:
                time.sleep(a.gap_ms / 1e3)

    for k in range(a.threads):
        threading.Thread(target=worker, args=(k,), name=f"load-{k}", daemon=True).start()
    t0 = time.time()
    prev = obs.snapshot({a.model: tok})
    try:
        while not a.duration or time.time() - t0 < a.duration:
            time.sleep(a.report_every or 1.0)
            if a.report_every:
                cur = obs.snapshot({a.model: tok})
                print(obs.report(prev, cur), flush=True)
                prev = cur
    except KeyboardInterrupt:
        pass
    stop.set()
    print(f"{done[0]} requests in {time.time() - t0:.0f} s")
    server.shutdown()


if __name__ == "__main__":
    main()
