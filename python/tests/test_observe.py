"""Tests of `fastokens.observe` and the counters behind it.

The first group runs on every build (the observe API must work, and say it is
off, without the `observe` feature). The rest needs a build with it
(`maturin build --features observe`) and checks that the counters agree
exactly with what the test did: each test thread has its own slot in the
snapshot, so per-thread counts are exact whatever else runs.
"""

import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest

import fastokens.observe as obs
from fastokens import _native
from fastokens._native import Tokenizer

MODEL = "Qwen/Qwen3-0.6B"
ENABLED = _native.observe_enabled()
needs_observe = pytest.mark.skipif(not ENABLED, reason="build without the observe feature")


def _tiny() -> Tokenizer:
    return Tokenizer.from_json_str(
        json.dumps(
            {
                "added_tokens": [
                    {"id": 5, "content": "<end>", "special": True, "single_word": False,
                     "lstrip": False, "rstrip": False, "normalized": False}
                ],
                "model": {"type": "BPE", "vocab": {"<": 0, "e": 1, "n": 2, "d": 3, ">": 4}, "merges": []},
            }
        )
    )


@pytest.fixture(scope="module")
def qwen():
    return Tokenizer.from_model(MODEL)


def _qwen_with_cache(mib: int) -> Tokenizer:
    old = os.environ.get("FASTOKENS_SEGMENT_CACHE")
    os.environ["FASTOKENS_SEGMENT_CACHE"] = str(mib)
    try:
        return Tokenizer.from_model(MODEL)
    finally:
        if old is None:
            del os.environ["FASTOKENS_SEGMENT_CACHE"]
        else:
            os.environ["FASTOKENS_SEGMENT_CACHE"] = old


def _chat(n_turns: int, seed: int = 0, body: int = 200) -> str:
    rng = random.Random(seed)
    words = ["read", "src/lib.rs", "fix", "the", "bug", "def", "encode(self):", "return", "ids", "\n", "    "]
    out = []
    for t in range(n_turns):
        role = "user" if t % 2 == 0 else "assistant"
        text = " ".join(rng.choice(words) for _ in range(body))
        out.append(f"<|im_start|>{role}\n{seed}.{t} {text}<|im_end|>\n")
    return "".join(out)


def _in_thread(name, fn):
    """Run `fn` on a new thread called `name`; returns its result."""
    box = {}

    def run():
        box["result"] = fn()

    t = threading.Thread(target=run, name=name)
    t.start()
    t.join()
    return box["result"]


def _slot(snap, name):
    slots = [t for t in snap["process"]["threads"] if t["name"] == name]
    assert len(slots) == 1, [t["name"] for t in snap["process"]["threads"]]
    return slots[0]


def _get(url):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.headers.get("Content-Type"), r.read()


# ── every build ──────────────────────────────────────────────────────────────


def test_snapshot_and_tokenizer_json_are_valid_in_every_build():
    tok = _tiny()
    p = json.loads(_native.observe_snapshot())
    assert p["enabled"] is ENABLED
    t = json.loads(tok.observe())
    assert t["enabled"] is ENABLED
    assert t["segment_cache"] is None
    assert t["prefix_cache"] is False
    s = obs.snapshot(tok)
    assert set(s) >= {"time", "pid", "cpu_s", "rss_bytes", "process", "tokenizers"}
    assert s["tokenizers"]["tokenizer"] == t
    assert obs.enabled() is ENABLED


def test_report_and_prometheus_work_on_any_snapshot():
    tok = _tiny()
    a = obs.snapshot({"tiny": tok})
    tok.encode("<end>end")
    b = obs.snapshot({"tiny": tok})
    text = obs.report(a, b)
    assert isinstance(text, str) and text
    if not ENABLED:
        assert "without the `observe` feature" in text
    prom = obs.prometheus(b)
    assert "fastokens_observe_enabled " + str(int(ENABLED)) in prom
    parser = pytest.importorskip("prometheus_client.parser")
    families = {f.name: f for f in parser.text_string_to_metric_families(prom)}
    assert "fastokens_process_cpu_seconds" in families or "fastokens_process_cpu_seconds_total" in families


def test_native_of_unwraps_and_rejects():
    tok = _tiny()

    class Shim:
        _fast = tok

    class Wrapper:
        _tokenizer = Shim()

    assert obs._native_of(tok) is tok
    assert obs._native_of(Shim()) is tok
    assert obs._native_of(Wrapper()) is tok
    with pytest.raises(TypeError):
        obs._native_of(object())
    with pytest.raises(TypeError):
        obs.serve({"bad": object()}, port=0)


def test_native_of_unwraps_a_patched_transformers_tokenizer():
    transformers = pytest.importorskip("transformers")
    import fastokens

    fastokens.patch_transformers()
    try:
        hf = transformers.AutoTokenizer.from_pretrained(MODEL)
        native = obs._native_of(hf)
        assert isinstance(native, Tokenizer)
        assert obs.snapshot({"hf": hf})["tokenizers"]["hf"]["enabled"] is ENABLED
    finally:
        fastokens.unpatch_transformers()


def test_http_server_serves_dashboard_json_and_metrics():
    tok = _tiny()
    server = obs.serve({"tiny": tok}, port=0)
    try:
        host, port = server.server_address[:2]
        base = f"http://{host}:{port}"
        status, ctype, body = _get(base + "/")
        assert status == 200 and ctype.startswith("text/html")
        assert b"<title>fastokens observe</title>" in body
        status, ctype, body = _get(base + "/json")
        assert status == 200 and ctype == "application/json"
        snap = json.loads(body)
        assert snap["process"]["enabled"] is ENABLED
        assert "tiny" in snap["tokenizers"]
        status, ctype, body = _get(base + "/metrics?x=1")
        assert status == 200 and ctype.startswith("text/plain")
        parser = pytest.importorskip("prometheus_client.parser")
        assert list(parser.text_string_to_metric_families(body.decode()))
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/nope")
        assert e.value.code == 404
        # Concurrent requests do not trip each other up.
        errors = []

        def hit():
            try:
                for _ in range(10):
                    json.loads(_get(base + "/json")[2])
            except Exception as ex:  # pragma: no cover - reported below
                errors.append(ex)

        ts = [threading.Thread(target=hit) for _ in range(8)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        assert not errors
    finally:
        server.shutdown()
        server.server_close()


def test_top_and_dump_commands_against_a_live_server():
    tok = _tiny()
    server = obs.serve({"tiny": tok}, port=0)
    try:
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}"
        top = subprocess.run([sys.executable, "-m", "fastokens.observe", "top", "--url", url, "--once", "--interval", "0.2"],
                             capture_output=True, text=True, timeout=60)
        assert top.returncode == 0, top.stderr
        assert ("API" in top.stdout) if ENABLED else ("observe" in top.stdout)
        dump = subprocess.run([sys.executable, "-m", "fastokens.observe", "dump", "--url", url],
                              capture_output=True, text=True, timeout=60)
        assert dump.returncode == 0, dump.stderr
        assert json.loads(dump.stdout)["process"]["enabled"] is ENABLED
    finally:
        server.shutdown()
        server.server_close()


def test_dashboard_script_is_valid_javascript(tmp_path):
    from fastokens._observe_html import DASHBOARD

    node = __import__("shutil").which("node")
    if not node:
        pytest.skip("node not installed")
    script = DASHBOARD.split("<script>")[1].split("</script>")[0]
    path = tmp_path / "dashboard.js"
    path.write_text(script)
    r = subprocess.run([node, "--check", str(path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


# ── observe builds ───────────────────────────────────────────────────────────


@needs_observe
def test_api_and_encode_counts_are_exact_per_thread(qwen):
    small = "hello world, this is a short request"
    big = _chat(30, seed=1)
    assert len(big.encode()) >= 16 * 1024 > len(small.encode())
    ids = qwen.encode(big).ids

    def work():
        for _ in range(3):
            qwen.encode(small)
        qwen.encode(big)
        qwen.encode_ordinary(small)
        # Two inputs each: rayon runs a one-input batch inline on the caller.
        qwen.encode_batch([small, big])
        qwen.encode_batch_flat([small, small])
        qwen.decode(ids)
        qwen.decode_batch([ids, ids[:10]])

    a = obs.snapshot()
    _in_thread("obs-exact", work)
    b = obs.snapshot()
    me = _slot(b, "py:obs-exact")
    assert me["alive"] is False
    assert me["api_calls"] == 9
    # encode, encode_ordinary run here; the batches run on rayon's pool.
    assert me["encodes"] == 5
    assert me["encode_bytes"] == 4 * len(small.encode()) + len(big.encode())
    api = lambda s, k, f: s["process"]["api"][k][f]
    d = lambda k, f: api(b, k, f) - api(a, k, f)
    assert d("encode", "calls") == 4
    assert d("encode", "gil_released") == 1  # only the >= 16 KB input
    assert d("encode", "units") == 3 * len(small.encode()) + len(big.encode())
    assert d("encode_ordinary", "calls") == 1
    assert d("encode_batch", "calls") == 1 and d("encode_batch", "gil_released") == 1
    assert d("encode_batch_flat", "calls") == 1
    assert d("decode", "calls") == 1 and d("decode", "units") == len(ids)
    assert d("decode_batch", "units") == len(ids) + 10
    assert sum(me["paths"].values()) == 5
    if json.loads(qwen.observe())["scanner_path"]:
        assert me["paths"]["scan_splits"] + me["paths"]["scan_whole"] == 5
    else:
        assert me["paths"]["fused_split"] + me["paths"]["general"] == 5
    jobs = lambda s: s["process"]["jobs"]["batch"]
    assert jobs(b)["jobs"] - jobs(a)["jobs"] == 2
    assert jobs(b)["tasks"] - jobs(a)["tasks"] == 4


@needs_observe
def test_gil_wait_is_counted_when_python_holds_the_gil(qwen):
    big = _chat(400, seed=2)  # a few ms of GIL-free work
    a = obs.snapshot()
    t = threading.Thread(target=lambda: [qwen.encode(big) for _ in range(5)], name="obs-gil")
    t.start()
    deadline = time.time() + 30
    x = 0
    while t.is_alive() and time.time() < deadline:
        for i in range(10_000):  # pure Python: holds the GIL between switches
            x += i
    t.join()
    b = obs.snapshot()
    d = b["process"]["api"]["encode"]["gil_wait_ns"] - a["process"]["api"]["encode"]["gil_wait_ns"]
    assert d > 0
    me = _slot(b, "py:obs-gil")
    assert me["api_ns"] >= me["encode_ns"] > 0


@needs_observe
def test_threads_are_named_and_marked_dead():
    tok = _tiny()
    _in_thread("obs-named", lambda: tok.encode("end"))
    me = _slot(obs.snapshot(), "py:obs-named")
    assert me["kind"] == "caller"
    assert me["alive"] is False
    assert me["os_tid"] > 0
    assert me["encodes"] == 1 and me["api_calls"] == 1


@needs_observe
def test_long_inputs_run_pool_jobs_on_named_workers(qwen):
    p = obs.snapshot()["process"]
    if p["pool"]["bpe_threads"] < 2:
        pytest.skip("single-threaded BPE pool")
    a = obs.snapshot()
    qwen.encode(_chat(2000, seed=3))  # several hundred KB of chat template
    plain = "".join(f"line {i}: the quick brown fox\n" for i in range(20_000))
    qwen.encode(plain)
    b = obs.snapshot()
    jobs = lambda s, k: s["process"]["jobs"][k]["jobs"]
    if json.loads(qwen.observe())["scanner_path"]:
        assert jobs(b, "chat_splits") > jobs(a, "chat_splits")
        assert jobs(b, "scan") > jobs(a, "scan")
    else:  # Split fused with the model: the splits are tokenized in parallel
        assert jobs(b, "pretokenized") >= jobs(a, "pretokenized") + 2
    workers = [t for t in b["process"]["threads"] if t["name"].startswith("fastokens-")]
    assert workers and all(t["kind"] == "worker" for t in workers)
    assert sum(t["tasks"] for t in workers) > 0


@needs_observe
def test_segment_cache_counters_follow_the_traffic():
    tok = _qwen_with_cache(64)
    chat = _chat(20, seed=4)
    first = tok.encode(chat).ids
    c1 = json.loads(tok.observe())["segment_cache"]
    second = tok.encode(chat).ids
    c2 = json.loads(tok.observe())["segment_cache"]
    assert first == second
    # The text segments between added tokens: the first request looks up each
    # distinct one once (a repeat within it reuses that lookup) and misses;
    # the second looks up every occurrence and hits.
    segs = [s.encode() for s in re.split(r"<\|im_start\|>|<\|im_end\|>", chat) if s]
    distinct = set(segs)
    assert (c1["hits"], c1["misses"]) == (0, len(distinct))
    assert c1["miss_bytes"] == sum(map(len, distinct))
    assert (c2["hits"], c2["misses"]) == (len(segs), len(distinct))
    assert c2["hit_bytes"] == sum(map(len, segs))
    assert c2["entries"] == len(distinct) and len(c2["shards"]) == c2["shard_count"] == 16
    assert sum(s["inserts"] for s in c2["shards"]) == len(distinct) - c2["rejected_too_large"]
    me_paths = _in_thread("obs-cache", lambda: (tok.encode(chat), obs.snapshot())[1])
    assert _slot(me_paths, "py:obs-cache")["paths"]["segments_cached"] == 1


@needs_observe
def test_a_small_cache_rotates_and_evicts():
    tok = _qwen_with_cache(1)  # 1 MiB: 32 KiB per shard's young generation
    for seed in range(60):
        tok.encode(_chat(10, seed=100 + seed, body=150))
    c = json.loads(tok.observe())["segment_cache"]
    rot = sum(s["rotations"] for s in c["shards"])
    ev = sum(s["evicted_entries"] for s in c["shards"])
    assert rot > 0 and ev > 0
    assert c["bytes"] <= c["approx_max_bytes"] * 1.5
    # Shards report generations that add up to the totals.
    assert sum(s["young_entries"] + s["old_entries"] for s in c["shards"]) == c["entries"]
    assert sum(s["young_bytes"] + s["old_bytes"] for s in c["shards"]) == c["bytes"]


@needs_observe
def test_write_lock_waits_are_counted_without_deadlock(qwen):
    big = _chat(1500, seed=5)
    before = obs.snapshot()["process"]["lock_waits"]["write"]["count"]
    stop = threading.Event()

    def readers():
        while not stop.is_set():
            qwen.encode(big)

    ts = [threading.Thread(target=readers, name=f"obs-reader-{i}") for i in range(3)]
    for t in ts:
        t.start()
    deadline = time.time() + 20
    try:
        while time.time() < deadline:
            qwen.enable_truncation(1_000_000)
            qwen.no_truncation()
            if obs.snapshot()["process"]["lock_waits"]["write"]["count"] > before:
                break
            time.sleep(0.001)
    finally:
        stop.set()
        for t in ts:
            t.join(timeout=30)
    assert not any(t.is_alive() for t in ts), "reader threads stuck"
    assert obs.snapshot()["process"]["lock_waits"]["write"]["count"] > before


@needs_observe
def test_counters_never_decrease_while_threads_work(qwen):
    chat = _chat(40, seed=6)
    stop = threading.Event()

    def worker():
        while not stop.is_set():
            qwen.encode(chat)
            qwen.encode_batch([chat, "x"])

    ts = [threading.Thread(target=worker, name=f"obs-mono-{i}") for i in range(4)]
    for t in ts:
        t.start()
    try:
        prev = obs.snapshot({"q": qwen})
        for _ in range(30):
            cur = obs.snapshot({"q": qwen})
            _assert_monotonic(prev["process"], cur["process"], "process")
            prev = cur
            time.sleep(0.01)
        assert "py:obs-mono-0" in obs.report(prev, obs.snapshot({"q": qwen}))
    finally:
        stop.set()
        for t in ts:
            t.join()


def _assert_monotonic(a, b, at):
    if isinstance(a, dict):
        for k, v in a.items():
            if k in ("alive", "kind", "name"):
                continue
            _assert_monotonic(v, b[k], f"{at}.{k}")
    elif isinstance(a, list):
        assert len(b) >= len(a), at
        for i, (x, y) in enumerate(zip(a, b)):
            _assert_monotonic(x, y, f"{at}[{i}]")
    elif isinstance(a, (int, float)) and not isinstance(a, bool):
        assert b >= a, f"{at}: {a} -> {b}"


@needs_observe
def test_prometheus_values_match_the_snapshot(qwen):
    qwen.encode(_chat(5, seed=7))
    parser = pytest.importorskip("prometheus_client.parser")
    snap = obs.snapshot({"q": qwen})
    samples = {}
    for fam in parser.text_string_to_metric_families(obs.prometheus(snap)):
        for s in fam.samples:
            samples[(s.name, tuple(sorted(s.labels.items())))] = s.value
    p = snap["process"]
    assert samples[("fastokens_api_calls_total", (("api", "encode"),))] == p["api"]["encode"]["calls"]
    assert samples[("fastokens_jobs_total", (("kind", "batch"),))] == p["jobs"]["batch"]["jobs"]
    count = samples[("fastokens_api_latency_seconds_count", (("api", "encode"),))]
    inf = samples[("fastokens_api_latency_seconds_bucket", (("api", "encode"), ("le", "+Inf")))]
    assert count == inf == p["api"]["encode"]["calls"]


@needs_observe
def test_report_mentions_what_happened(qwen):
    a = obs.snapshot({"q": qwen})
    _in_thread("obs-report", lambda: qwen.encode(_chat(10, seed=8)))
    b = obs.snapshot({"q": qwen})
    text = obs.report(a, b)
    assert "encode" in text and "py:obs-report" in text
