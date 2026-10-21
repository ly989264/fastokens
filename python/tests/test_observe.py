"""Tests of `fastokens.observe` and the counters behind it.

The first group runs on every build (the observe API must work, and say it is
off, without the `observe` feature). The rest needs a build with it
(`maturin build --features observe`) and checks that the counters and the
analysis agree exactly with what the test did: each test thread has its own
slot in the snapshot, and pytest runs tests one at a time, so per-thread and
per-window counts are exact.
"""

import base64
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


def _with_env(name, value, make):
    old = os.environ.get(name)
    os.environ[name] = str(value)
    try:
        return make()
    finally:
        if old is None:
            del os.environ[name]
        else:
            os.environ[name] = old


def _scanner(tmp_path, **env) -> Tokenizer:
    """A byte-level tiktoken model with the o200k pattern: the scanner path."""
    ranks = [bytes([i]) for i in range(256)] + [b"th", b"the", b" the", b"in", b"ing", b" th", b"he", b"ll", b"hell", b"hello"]
    path = tmp_path / "scan.model"
    path.write_text("".join(f"{base64.b64encode(t).decode()} {i}\n" for i, t in enumerate(ranks)))

    def make():
        return Tokenizer.from_tiktoken(str(path), encoding="o200k_base",
                                       special_tokens={"<|im_start|>": 400, "<|im_end|>": 401})

    for k, v in env.items():
        make = (lambda m, k, v: lambda: _with_env(k, v, m))(make, k, v)
    tok = make()
    assert json.loads(tok.observe())["scanner_path"] is True
    return tok


@pytest.fixture(scope="module")
def qwen():
    return Tokenizer.from_model(MODEL)


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


def _post(url, body=b""):
    req = urllib.request.Request(url, data=body, method="POST")
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.status, r.read()


# ── every build ──────────────────────────────────────────────────────────────


def test_snapshot_and_tokenizer_json_are_valid_in_every_build(monkeypatch):
    # The caches are off unless the environment turns them on.
    monkeypatch.delenv("FASTOKENS_INPUT_CACHE", raising=False)
    monkeypatch.delenv("FASTOKENS_SEGMENT_CACHE", raising=False)
    tok = _tiny()
    p = json.loads(_native.observe_snapshot())
    assert p["enabled"] is ENABLED
    t = json.loads(tok.observe())
    assert t["enabled"] is ENABLED
    assert t["segment_cache"] is None
    assert t["prefix_cache"] is None
    s = obs.snapshot(tok)
    assert set(s) >= {"time", "pid", "cpu_s", "rss_bytes", "process", "tokenizers"}
    assert s["tokenizers"]["tokenizer"] == t
    assert obs.enabled() is ENABLED
    obs.reset_slow()  # a no-op without the feature


def test_report_analyze_and_prometheus_work_on_any_snapshot():
    tok = _tiny()
    with obs.profile({"tiny": tok}) as p:
        tok.encode("<end>end")
    text = p.report()
    assert isinstance(text, str) and text
    assert p.data["enabled"] is ENABLED
    if not ENABLED:
        assert "without the `observe` feature" in text
    prom = obs.prometheus(p.after)
    assert "fastokens_observe_enabled " + str(int(ENABLED)) in prom
    parser = pytest.importorskip("prometheus_client.parser")
    families = {f.name for f in parser.text_string_to_metric_families(prom)}
    assert any(f.startswith("fastokens_process_cpu_seconds") for f in families)


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
    with pytest.raises(TypeError):
        obs.profile({"bad": object()})


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


@pytest.mark.skipif(not ENABLED, reason="needs the observe build")
def test_the_dashboard_lists_each_windows_own_slowest_encodes():
    tok = _tiny()
    obs.reset_slow()
    # Fill the since-reset list with slow encodes: it then turns away the
    # quick ones below, which the window must still list.
    for _ in range(40):
        tok.encode("end" * 20000)
    time.sleep(1.05)  # the next encodes run in a second of their own
    s0 = obs.snapshot()
    for _ in range(5):
        tok.encode("end<end>end")
    s1 = obs.snapshot()
    assert not obs.analyze(s0, s1)["slowest"]
    slow = obs._analyze_window(s0, s1)["slowest"]
    assert len(slow) == 5
    assert all(s0["process"]["uptime_s"] < s["at_s"] <= s1["process"]["uptime_s"] for s in slow)
    assert [s["bytes"] for s in slow] == [len("end<end>end")] * 5


def test_http_server_serves_dashboard_json_metrics_and_posts():
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
        # /analyze: the first call has nothing to compare with, the second
        # analyzes the window since the snapshot it sends.
        status, body = _post(base + "/analyze")
        first = json.loads(body)
        assert first["analysis"] is None
        tok.encode("end<end>")
        status, body = _post(base + "/analyze", json.dumps(first["snapshot"]).encode())
        second = json.loads(body)
        expected = obs.analyze(first["snapshot"], second["snapshot"])
        got = second["analysis"]
        # The window's own slowest encodes, beyond those analyze() sees.
        if ENABLED:
            t0, t1 = first["snapshot"]["process"]["uptime_s"], second["snapshot"]["process"]["uptime_s"]
            slow = got.pop("slowest")
            ns = [s["ns"] for s in slow]
            assert ns and ns == sorted(ns, reverse=True)
            assert all(t0 < s["at_s"] <= t1 for s in slow)
            expected.pop("slowest")
        assert got == expected
        # Several windows in one call, by label.
        status, body = _post(base + "/analyze", json.dumps({"windows": {"1": second["snapshot"], "10": first["snapshot"]}}).encode())
        both = json.loads(body)
        assert set(both["analyses"]) == {"1", "10"}
        if ENABLED:
            assert both["analyses"]["10"]["window_s"] > both["analyses"]["1"]["window_s"]
            assert both["analyses"]["10"]["paths"]
        assert _post(base + "/reset-slow")[0] == 200
        with pytest.raises(urllib.error.HTTPError) as e:
            _get(base + "/nope")
        assert e.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/nope")
        assert e.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/analyze", b"{not json")
        assert e.value.code == 500
        # Concurrent clients do not trip each other up (the default listen
        # backlog of 5 reset connections here).
        errors = []

        def hit():
            try:
                for _ in range(10):
                    json.loads(_get(base + "/json")[2])
                    _post(base + "/analyze", json.dumps(first["snapshot"]).encode())
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


def test_top_profile_and_dump_commands_against_a_live_server():
    tok = _tiny()
    server = obs.serve({"tiny": tok}, port=0)
    stop = threading.Event()

    def load():
        while not stop.is_set():
            tok.encode("end<end>end")
            time.sleep(0.001)

    loader = threading.Thread(target=load, name="obs-cli-load")
    loader.start()
    try:
        host, port = server.server_address[:2]
        url = f"http://{host}:{port}"

        def run(*args):
            return subprocess.run([sys.executable, "-m", "fastokens.observe", *args, "--url", url],
                                  capture_output=True, text=True, timeout=60)

        top = run("top", "--once", "--interval", "0.3")
        assert top.returncode == 0, top.stderr
        prof = run("profile", "--seconds", "0.5")
        assert prof.returncode == 0, prof.stderr
        prof_json = run("profile", "--seconds", "0.5", "--json")
        assert prof_json.returncode == 0, prof_json.stderr
        data = json.loads(prof_json.stdout)
        assert data["enabled"] is ENABLED
        if ENABLED:
            assert "where encode time goes" in top.stdout
            assert "where encode time goes" in prof.stdout and "slowest encodes" in prof.stdout
            assert data["encodes"]["total"] > 0
            assert data["slowest"] and all(s["thread"] == "py:obs-cli-load" for s in data["slowest"])
        else:
            assert "observe" in top.stdout
        dump = run("dump")
        assert dump.returncode == 0, dump.stderr
        assert json.loads(dump.stdout)["process"]["enabled"] is ENABLED
    finally:
        stop.set()
        loader.join()
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
        n_tokens = 0
        for _ in range(3):
            n_tokens += len(qwen.encode(small))
        n_tokens += len(qwen.encode(big))
        n_tokens += len(qwen.encode_ordinary(small))
        # Two inputs each: rayon runs a one-input batch inline on the caller.
        qwen.encode_batch([small, big])
        qwen.encode_batch_flat([small, small])
        qwen.decode(ids)
        qwen.decode_batch([ids, ids[:10]])
        return n_tokens

    a = obs.snapshot()
    n_tokens = _in_thread("obs-exact", work)
    b = obs.snapshot()
    me = _slot(b, "py:obs-exact")
    assert me["alive"] is False
    assert me["api_calls"] == 9
    # encode and encode_ordinary run here; the batches run on rayon's pool.
    assert me["encodes"] == 5
    assert me["encode_bytes"] == 4 * len(small.encode()) + len(big.encode())
    assert me["encode_tokens"] == n_tokens
    api = lambda s, k, f: s["process"]["api"][k][f]
    d = lambda k, f: api(b, k, f) - api(a, k, f)
    assert d("encode", "calls") == 4
    assert d("encode", "gil_released") == 1  # only the >= 16 KB input
    assert d("encode", "units") == 3 * len(small.encode()) + len(big.encode())
    assert 0 < d("encode", "rust_ns") <= d("encode", "ns")
    assert d("encode_ordinary", "calls") == 1
    assert d("encode_batch", "calls") == 1 and d("encode_batch", "gil_released") == 1
    assert d("encode_batch_flat", "calls") == 1
    assert d("decode", "calls") == 1 and d("decode", "units") == len(ids)
    assert 0 < d("decode", "rust_ns") <= d("decode", "ns")
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
def test_reading_encoding_lists_is_counted(qwen):
    enc = qwen.encode(_chat(3, seed=9))
    a = obs.snapshot()["process"]["api"]
    ids = enc.ids
    mask = enc.attention_mask
    b = obs.snapshot()["process"]["api"]
    assert b["encoding_ids"]["calls"] - a["encoding_ids"]["calls"] == 1
    assert b["encoding_ids"]["units"] - a["encoding_ids"]["units"] == len(ids)
    assert b["encoding_lists"]["calls"] - a["encoding_lists"]["calls"] == 1
    assert b["encoding_lists"]["units"] - a["encoding_lists"]["units"] == len(mask)


@needs_observe
def test_profile_breaks_encodes_into_stages(tmp_path, qwen):
    scan = _scanner(tmp_path)
    chats = [_chat(6, seed=s) for s in range(5)]
    tokens = 0
    with obs.profile({"scan": scan, "qwen": qwen}) as p:
        for c in chats:
            tokens += len(scan.encode(c).ids)
            tokens += len(qwen.encode(c).ids)
    a = p.data
    assert a["encodes"]["total"] == 10
    assert sum(x["tokens"] for x in a["paths"].values()) == tokens
    assert sum(x["bytes"] for x in a["paths"].values()) == 2 * sum(len(c.encode()) for c in chats)
    s = a["paths"]["scan_splits"]
    assert s["encodes"] == 5
    assert s["stage_ms"]["added_tokens"] > 0 and s["stage_ms"]["model"] > 0
    assert s["stage_ms"]["pretokenize"] == 0  # fused into the scan
    q = a["paths"]["fused_split"]
    assert q["stage_ms"]["pretokenize"] > 0 and q["stage_ms"]["model"] > 0
    for path in a["paths"].values():
        assert sum(path["stage_ms"].values()) <= path["mean_ms"] * (1 + 1e-9)
        assert path["other_ms"] >= 0
        assert path["bytes_per_token"] > 0
    # Every encode is among the slowest kept (fewer than 32 ran).
    assert len(a["slowest"]) == 10
    ns = [x["ns"] for x in a["slowest"]]
    assert ns == sorted(ns, reverse=True)
    assert all(sum(x["stage_ns"].values()) <= x["ns"] for x in a["slowest"])
    text = p.report()
    assert "where encode time goes" in text and "scan_splits" in text and "fused_split" in text
    assert "slowest encodes" in text and "BPE pretoken cache" in text


@needs_observe
def test_bpe_cache_hits_on_a_repeat(tmp_path, monkeypatch):
    # A prefix cache would answer the repeat without the BPE cache.
    monkeypatch.delenv("FASTOKENS_INPUT_CACHE", raising=False)
    scan = _scanner(tmp_path)
    text = " ".join(f"word{i}" for i in range(2000))  # under 64 KiB: runs here
    with obs.profile() as first:
        scan.encode(text)
    with obs.profile() as second:
        scan.encode(text)
    f1, f2 = first.data["bpe"]["fused"], second.data["bpe"]["fused"]
    assert f1["lookups"] == f2["lookups"] >= 2000
    assert f1["misses"] > 0 and f1["merge_ms"] > 0
    assert f2["misses"] == 0 and f2["hit_rate"] == 1.0
    long_word = "x" * 64
    with obs.profile() as third:
        scan.encode(long_word)
        scan.encode(long_word)
    f3 = third.data["bpe"]["fused"]
    assert f3["misses"] == f3["uncacheable"] == 2  # no inline key: merged each time


@needs_observe
def test_prefix_cache_reuse_is_counted(tmp_path):
    tok = _scanner(tmp_path, FASTOKENS_INPUT_CACHE=4)
    plain = _scanner(tmp_path)
    base = "".join(f"line {i}: hello the thing\n" for i in range(1500))
    longer = base + "hello the end\n"
    with obs.profile({"t": tok}) as p:
        for text in (base, base, longer):
            assert tok.encode(text).ids == plain.encode(text).ids
    pc = p.data["tokenizers"]["t"]["prefix_cache"]
    assert (pc["lookups"], pc["exact_hits"], pc["prefix_hits"], pc["misses"]) == (3, 1, 1, 1)
    assert pc["reused_tokens"] >= 256
    assert 0.5 < pc["reused_share"] < 1
    assert p.data["paths"]["scan_whole"]["stage_ms"]["prefix_cache"] > 0
    assert "prefix cache [t]" in p.report()


@needs_observe
def test_chat_prefix_cache_counters_reconcile(tmp_path):
    tok = _scanner(tmp_path, FASTOKENS_INPUT_CACHE=4)
    plain = _scanner(tmp_path)
    base = _chat(12, seed=11)
    longer = base + _chat(1, seed=12)
    short = _chat(1, seed=13, body=10)
    prose = "".join(f"line {i}: hello the thing\n" for i in range(400))
    assert len(short) < 4096 < len(base) and len(prose) > 4096
    with obs.profile({"t": tok}) as p:
        # A miss, an exact repeat, a continuation; then input the chat path
        # leaves alone: too short, and without added tokens.
        for text in (base, base, longer, short, prose):
            assert tok.encode(text).ids == plain.encode(text).ids
    c = json.loads(tok.observe())["prefix_cache"]["chat"]
    assert (c["lookups"], c["exact_hits"], c["prefix_hits"], c["misses"]) == (3, 1, 1, 1)
    assert (c["bypass_short"], c["bypass_plain"], c["marker_fallbacks"]) == (1, 1, 0)
    assert (c["inserts"], c["replaced"], c["evicted_entries"], c["entries"]) == (2, 1, 0, 1)
    assert (c["cut_backoffs"], c["no_cut"], c["lock_contended"]) == (0, 0, 0)
    # Invariants: every looked-up byte was reused or encoded; every insert is
    # still there, was replaced or was evicted.
    assert c["lookup_bytes"] == 2 * len(base) + len(longer)
    assert c["reused_bytes"] + c["encoded_bytes"] == c["lookup_bytes"]
    assert c["reused_bytes"] > len(base) and c["reused_tokens"] > 0
    assert c["entries"] == c["inserts"] - c["replaced"] - c["evicted_entries"] - c["cleared_entries"]
    assert c["bytes"] > len(longer) and c["index_blocks"] >= len(longer) // 4096
    assert (c["rejected_oversized"], c["max_bytes"]) == (0, 2048 * 2**20)
    # The totals under `prefix_cache` add the plain-text cache's (the prose).
    top = json.loads(tok.observe())["prefix_cache"]
    assert top["lookups"] == c["lookups"] + 1
    # Analysis, report (`top`) and Prometheus agree with the counters.
    a = p.data["tokenizers"]["t"]["prefix_cache"]["chat"]
    assert (a["lookups"], a["exact_hits"], a["prefix_hits"], a["replaced"]) == (3, 1, 1, 1)
    assert a["reused_share"] == pytest.approx(c["reused_bytes"] / c["lookup_bytes"])
    report = p.report()
    assert "prefix cache (chat) [t]" in report and "cut backoffs 0" in report and "rejected too large 0" in report
    parser = pytest.importorskip("prometheus_client.parser")
    samples = {}
    for fam in parser.text_string_to_metric_families(obs.prometheus(obs.snapshot({"t": tok}))):
        for s in fam.samples:
            samples[s.name] = s.value
    assert samples["fastokens_prefix_cache_chat_lookups_total"] == c["lookups"]
    assert samples["fastokens_prefix_cache_chat_replaced_total"] == c["replaced"]
    assert samples["fastokens_prefix_cache_chat_bytes"] == c["bytes"]
    assert samples["fastokens_prefix_cache_chat_max_bytes"] == c["max_bytes"]


@needs_observe
def test_chat_prefix_cache_keeps_to_its_byte_budget(tmp_path):
    # 2 MiB: an input whose entry would cost over 32 KiB is not cached (this
    # model has about one token per byte, so an entry costs ~5 bytes a byte).
    tok = _scanner(tmp_path, FASTOKENS_INPUT_CACHE=4, FASTOKENS_INPUT_CACHE_MIB=2)
    plain = _scanner(tmp_path)
    small, big = _chat(4, seed=21), _chat(40, seed=22)
    assert 4096 < len(small) < 7000 and len(big) > 20_000
    for text in (small, big, big):
        assert tok.encode(text).ids == plain.encode(text).ids
    c = json.loads(tok.observe())["prefix_cache"]["chat"]
    assert c["max_bytes"] == 2 * 2**20
    assert (c["inserts"], c["entries"], c["rejected_oversized"]) == (1, 1, 2)
    assert c["rejected_bytes"] > 2 * len(big)


@needs_observe
def test_segment_cache_counters_follow_the_traffic(monkeypatch):
    # A prefix cache would answer the repeat before the segment cache.
    monkeypatch.delenv("FASTOKENS_INPUT_CACHE", raising=False)
    tok = _with_env("FASTOKENS_SEGMENT_CACHE", 64, lambda: Tokenizer.from_model(MODEL))
    chat = _chat(20, seed=4)
    first = tok.encode(chat).ids
    c1 = json.loads(tok.observe())["segment_cache"]
    with obs.profile({"q": tok}) as p:
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
    seg = p.data["tokenizers"]["q"]["segment_cache"]
    assert seg["hit_rate"] == 1.0 and seg["byte_hit_rate"] == 1.0
    assert seg["miss_tokenize_ms"] == 0 and seg["est_saved_ms"] is None  # no misses to price hits by
    stages = p.data["paths"]["segments_cached"]["stage_ms"]
    assert stages["cache_lookup"] > 0 and stages["model"] == 0  # all hits: no model work
    me_paths = _in_thread("obs-cache", lambda: (tok.encode(chat), obs.snapshot())[1])
    assert _slot(me_paths, "py:obs-cache")["paths"]["segments_cached"] == 1


@needs_observe
def test_segment_cache_saving_is_estimated_from_misses():
    tok = _with_env("FASTOKENS_SEGMENT_CACHE", 64, lambda: Tokenizer.from_model(MODEL))
    with obs.profile({"q": tok}) as p:
        for seed in range(5):
            tok.encode(_chat(10, seed=200 + seed))
            tok.encode(_chat(10, seed=200 + seed))
    seg = p.data["tokenizers"]["q"]["segment_cache"]
    assert 0 < seg["hit_rate"] < 1
    assert seg["miss_tokenize_ms"] > 0 and seg["est_saved_ms"] > 0


@needs_observe
def test_a_small_cache_rotates_and_evicts():
    tok = _with_env("FASTOKENS_SEGMENT_CACHE", 1, lambda: Tokenizer.from_model(MODEL))
    for seed in range(60):
        tok.encode(_chat(10, seed=100 + seed, body=150))
    c = json.loads(tok.observe())["segment_cache"]
    rot = sum(s["rotations"] for s in c["shards"])
    ev = sum(s["evicted_entries"] for s in c["shards"])
    assert rot > 0 and ev > 0
    assert c["bytes"] <= c["approx_max_bytes"] * 1.5
    assert sum(s["young_entries"] + s["old_entries"] for s in c["shards"]) == c["entries"]
    assert sum(s["young_bytes"] + s["old_bytes"] for s in c["shards"]) == c["bytes"]


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
def test_long_inputs_run_pool_jobs_on_named_workers(qwen):
    p = obs.snapshot()["process"]
    if p["pool"]["bpe_threads"] < 2:
        pytest.skip("single-threaded BPE pool")
    with obs.profile({"q": qwen}) as prof:
        qwen.encode(_chat(2000, seed=3))  # several hundred KB of chat template
        plain = "".join(f"line {i}: the quick brown fox\n" for i in range(20_000))
        qwen.encode(plain)
    jobs = prof.data["jobs"]
    if json.loads(qwen.observe())["scanner_path"]:
        assert jobs["chat_splits"]["jobs"] >= 1 and jobs["scan"]["jobs"] >= 1
    else:  # Split fused with the model: the splits are tokenized in parallel
        assert jobs["pretokenized"]["jobs"] >= 2
    # Tasks ran and were timed; how many ran at once depends on free cores.
    assert all(j["parallelism"] and j["parallelism"] > 0 for j in jobs.values())
    workers = [t for t in prof.after["process"]["threads"] if t["name"].startswith("fastokens-")]
    assert workers and all(t["kind"] == "worker" for t in workers)
    assert sum(t["tasks"] for t in workers) > 0


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
            if k in ("alive", "kind", "name", "slowest", "stages"):
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
    assert samples[("fastokens_bpe_fused_lookups_total", ())] == p["bpe"]["fused_lookups"]
    path = "fused_split"
    assert samples[("fastokens_encode_tokens_total", (("path", path),))] == p["paths"][path]["tokens"]
    stage = samples[("fastokens_encode_stage_seconds_total", (("path", path), ("stage", "model")))]
    assert stage == pytest.approx(p["paths"][path]["stage_ns"]["model"] / 1e9)
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
