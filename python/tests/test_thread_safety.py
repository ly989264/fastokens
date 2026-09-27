"""Concurrency tests for a `Tokenizer` shared between Python threads.

Reads (encode/decode) hold the state's read lock and mutators
(`enable_truncation`, `enable_padding`, …) its write lock. Encoding a large
input releases the GIL, so other threads keep running while it works; the read
lock is always dropped before the GIL is re-acquired, otherwise a mutator that
holds the GIL while waiting for the write lock would deadlock the process.
"""

import random
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from fastokens._native import Tokenizer

MODEL = "Qwen/Qwen3-0.6B"


def _large_text(seed: int) -> str:
    """~1.5 MB of varied code-like text with some CJK: tens of milliseconds to
    encode, and distinct per seed so no whole-input cache makes it cheaper."""
    rng = random.Random(seed)
    words = ["fn", "encode", "self", "input", "Vec<u32>", "tokenizer", "Hello,", "world!",
             "这是", "一个", "测试", "return", "match", "Some(x)", "=>", "{", "}", "0x1f"]
    lines = []
    for _ in range(25_000):
        lines.append(" ".join(rng.choice(words) + str(rng.randint(0, 999)) for _ in range(8)))
    return "\n".join(lines)


def test_decode_and_enable_truncation_concurrent():
    tok = Tokenizer.from_model(MODEL)
    ids = tok.encode("Hello, world! This is a thread-safety smoke test.").ids
    assert ids, "needed non-empty ids to decode"

    stop = threading.Event()
    errors: list[BaseException] = []

    def reader() -> None:
        try:
            while not stop.is_set():
                tok.decode(ids)
        except BaseException as exc:
            errors.append(exc)

    def writer() -> None:
        try:
            i = 0
            while not stop.is_set():
                if i % 2 == 0:
                    tok.enable_truncation(max_length=8)
                else:
                    tok.no_truncation()
                i += 1
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(4)] + [
        threading.Thread(target=writer) for _ in range(2)
    ]

    for t in threads:
        t.start()
    time.sleep(0.5)
    stop.set()
    for t in threads:
        t.join(timeout=5)
        if t.is_alive():
            pytest.fail(f"thread {t.name} did not exit")

    assert not errors, f"got {len(errors)} error(s); first: {errors[0]!r}"


@pytest.mark.parametrize(
    "method",
    ["encode", "encode_ordinary", "encode_segments", "encode_batch", "encode_batch_flat"],
)
def test_large_encode_releases_the_gil(method):
    """Another Python thread keeps running while a large input is encoded.

    With the GIL held for the whole call, a ticker thread gets at most one or
    two turns inside the call window; with it released, it ticks throughout.
    """
    tok = Tokenizer.from_model(MODEL)
    text = _large_text(seed=1)
    call = {
        "encode": lambda: tok.encode(text),
        "encode_ordinary": lambda: tok.encode_ordinary(text),
        "encode_segments": lambda: tok.encode_segments([(text, True)]),
        "encode_batch": lambda: tok.encode_batch([text]),
        "encode_batch_flat": lambda: tok.encode_batch_flat([text]),
    }[method]
    tok.encode_batch([_large_text(seed=0)])  # warm word caches on other text

    ticks: list[float] = []
    stop = threading.Event()

    def ticker() -> None:
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.0005)

    th = threading.Thread(target=ticker)
    th.start()
    try:
        time.sleep(0.01)
        start = time.perf_counter()
        call()
        end = time.perf_counter()
    finally:
        stop.set()
        th.join()

    inside = sum(start < t < end for t in ticks)
    assert inside >= 5, (
        f"{method} blocked other threads: {inside} ticks during a "
        f"{(end - start) * 1e3:.1f} ms call"
    )


def test_encode_and_mutators_do_not_deadlock():
    """Encodes that release the GIL race mutators that hold it.

    A deadlock here freezes the whole interpreter (the blocked mutator holds
    the GIL), so the race runs in a subprocess under a timeout.
    """
    script = textwrap.dedent(
        f"""
        import threading, time
        from fastokens._native import Tokenizer

        tok = Tokenizer.from_model({MODEL!r})
        text = {_large_text(seed=2)[:400_000]!r}
        stop = threading.Event()

        calls = [
            lambda: tok.encode(text),
            lambda: tok.encode_ordinary(text),
            lambda: tok.encode_segments([(text, True), (text, False)]),
            lambda: tok.encode_batch([text, text]),
            lambda: tok.encode_batch_flat([text, text]),
        ]

        def encoder(call):
            while not stop.is_set():
                call()

        def mutator():
            i = 0
            while not stop.is_set():
                if i % 4 == 0:
                    tok.enable_truncation(max_length=10**9)
                elif i % 4 == 1:
                    tok.no_truncation()
                elif i % 4 == 2:
                    tok.enable_padding(pad_id=0)
                else:
                    tok.no_padding()
                i += 1

        threads = [threading.Thread(target=encoder, args=(c,)) for c in calls]
        threads += [threading.Thread(target=mutator) for _ in range(2)]
        for t in threads:
            t.start()
        time.sleep(2)
        stop.set()
        for t in threads:
            t.join()
        print("ok")
        """
    )
    try:
        result = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
        )
    except subprocess.TimeoutExpired:
        pytest.fail("encode racing enable_truncation/enable_padding deadlocked")
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
