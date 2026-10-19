"""Race each encode method that may release the GIL against mutators that take
the write lock while holding the GIL, one method per subprocess, under a
timeout; plus a GIL-release check per method (does a ticker thread run during a
large encode). usage: python deadlock_repro.py [--seconds 3] [--timeout 60]"""
import argparse
import subprocess
import sys

CHILD = r'''
import random, sys, threading, time
from fastokens._native import Tokenizer

def large_text(seed, n=25_000):
    rng = random.Random(seed)
    words = ["fn", "encode", "self", "input", "Vec<u32>", "tokenizer", "Hello,", "world!",
             "这是", "一个", "测试", "return", "match", "Some(x)", "=>", "{", "}", "0x1f"]
    return "\n".join(" ".join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(8)) for _ in range(n))

mode, method, seconds = sys.argv[1], sys.argv[2], float(sys.argv[3])
tok = Tokenizer.from_model("Qwen/Qwen3-0.6B")
text = large_text(2)[:400_000] if mode == "race" else large_text(1)
call = {
    "encode": lambda: tok.encode(text),
    "encode_ordinary": lambda: tok.encode_ordinary(text),
    "encode_segments": lambda: tok.encode_segments([(text, True), (text, False)]),
    "encode_batch": lambda: tok.encode_batch([text, text]),
    "encode_batch_flat": lambda: tok.encode_batch_flat([text, text]),
}[method]

if mode == "race":
    stop = threading.Event(); n = [0]
    def encoder():
        while not stop.is_set():
            call(); n[0] += 1
    muts = [lambda: tok.enable_truncation(max_length=10**9), tok.no_truncation,
            lambda: tok.enable_padding(pad_id=0), tok.no_padding]
    def mutator():
        i = 0
        while not stop.is_set():
            muts[i % 4](); i += 1
    ths = [threading.Thread(target=encoder) for _ in range(2)] + [threading.Thread(target=mutator) for _ in range(2)]
    for t in ths: t.start()
    time.sleep(seconds); stop.set()
    for t in ths: t.join()
    print("ok", n[0])
else:
    tok.encode_batch([large_text(0)])
    ticks = []; stop = threading.Event()
    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter()); time.sleep(0.0005)
    th = threading.Thread(target=ticker); th.start(); time.sleep(0.01)
    t0 = time.perf_counter(); call(); t1 = time.perf_counter()
    stop.set(); th.join()
    print(sum(t0 < t < t1 for t in ticks), round((t1 - t0) * 1e3, 1))
'''

METHODS = ["encode", "encode_ordinary", "encode_segments", "encode_batch", "encode_batch_flat"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=3)
    ap.add_argument("--timeout", type=float, default=60)
    a = ap.parse_args()
    for m in METHODS:
        try:
            r = subprocess.run([sys.executable, "-c", CHILD, "race", m, str(a.seconds)],
                               capture_output=True, text=True, timeout=a.timeout)
            race = r.stdout.strip() if r.returncode == 0 else f"error {r.returncode}: {r.stderr.strip().splitlines()[-1:]}"
        except subprocess.TimeoutExpired:
            race = f"DEADLOCK (no exit within {a.timeout:.0f} s)"
        r = subprocess.run([sys.executable, "-c", CHILD, "tick", m, "0"], capture_output=True, text=True, timeout=300)
        ticks = r.stdout.strip().split() if r.returncode == 0 else ["?", "?"]
        print(f"{m:18} race vs mutators: {race:40} ticker during a {ticks[1]} ms call: {ticks[0]} ticks", flush=True)


main()
