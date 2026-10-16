"""Profile a running fastokens from the inside: how long each stage of the
encode pipeline takes on each path, how well each internal cache works (the
BPE word caches, the segment cache, the prefix cache), the thread pools and
the threads doing the work, the slowest requests with their stage breakdown,
and the Python API calls split into Rust, GIL wait and Python time.

Needs a build with the ``observe`` cargo feature; without it every counter is
absent and :func:`enabled` is ``False``::

    maturin build --release --features observe

Profile a block of code::

    import fastokens.observe as obs
    with obs.profile({"glm": tok}) as p:
        ...                                # the workload
    print(p.report())                      # text; p.data is the same as a dict

Or watch a running process (a server, a benchmark)::

    obs.serve({"glm": tok}, port=9464)     # dashboard at http://127.0.0.1:9464
                                           # plus /json and /metrics (Prometheus)

and from another terminal::

    python -m fastokens.observe top     --url http://127.0.0.1:9464
    python -m fastokens.observe profile --url http://127.0.0.1:9464 --seconds 10

``tok`` may be a ``fastokens._native.Tokenizer``, the shim that
``fastokens.patch_transformers()`` installs, or a ``transformers`` fast
tokenizer wrapping either.

Counters only grow; everything is computed from the difference of two
snapshots (:func:`analyze`). Histograms are log2 buckets: bucket ``i`` counts
values in ``[2**(i-1), 2**i)`` (microseconds or bytes), bucket 0 zeros.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import _native

__all__ = ["enabled", "snapshot", "analyze", "report", "profile", "reset_slow", "serve", "prometheus"]

STAGES = ["added_tokens", "normalize", "prefix_cache", "pretokenize", "model",
          "cache_lookup", "cache_assemble", "post_process"]


def enabled() -> bool:
    """Whether this build has the ``observe`` feature."""
    return _native.observe_enabled()


def reset_slow() -> None:
    """Forget the slowest encodes kept so far (start a fresh window)."""
    _native.observe_reset_slow()


def _native_of(tok):
    """The native tokenizer behind ``tok`` (a shim or transformers wrapper)."""
    for _ in range(5):
        if isinstance(tok, _native.Tokenizer):
            return tok
        for attr in ("_fast", "_tokenizer", "backend_tokenizer", "tokenizer"):
            inner = getattr(tok, attr, None)
            if inner is not None and inner is not tok:
                tok = inner
                break
        else:
            break
    raise TypeError(f"no fastokens tokenizer inside {type(tok).__name__}")


def _named(tokenizers):
    if tokenizers is None:
        return {}
    if not isinstance(tokenizers, dict):
        tokenizers = {"tokenizer": tokenizers}
    return {name: _native_of(t) for name, t in tokenizers.items()}


def _rss_bytes() -> int | None:
    try:
        import psutil

        return psutil.Process().memory_info().rss
    except Exception:
        pass
    try:
        with open("/proc/self/statm") as f:
            return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")
    except Exception:
        return None


def _cpu_s() -> float:
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF)
        return r.ru_utime + r.ru_stime
    except Exception:
        return time.process_time()


def snapshot(tokenizers=None) -> dict:
    """Process-wide counters plus each named tokenizer's (``{name: tok}``, or
    a single tokenizer, named ``"tokenizer"``)."""
    natives = _named(tokenizers)
    return {
        # Taken together, so CPU per wall second is right for short windows.
        "time": time.time(),
        "cpu_s": _cpu_s(),
        "pid": os.getpid(),
        "rss_bytes": _rss_bytes(),
        "process": json.loads(_native.observe_snapshot()),
        "tokenizers": {name: json.loads(t.observe()) for name, t in natives.items()},
    }


# ── analysis of a window ──────────────────────────────────────────────────────


def _d(a, b):
    """b - a for numbers and numeric lists, recursively for dicts."""
    if isinstance(b, dict):
        return {k: _d(a.get(k) if isinstance(a, dict) else None, v) for k, v in b.items()}
    if isinstance(b, list) and b and all(isinstance(x, (int, float)) for x in b):
        a = a if isinstance(a, list) and len(a) == len(b) else [0] * len(b)
        return [y - x for x, y in zip(a, b)]
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return b - (a if isinstance(a, (int, float)) else 0)
    return b


def _pct(hist, q):
    """Upper bound of the log2 bucket holding quantile ``q`` (None if empty)."""
    total = sum(hist)
    if not total:
        return None
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= q * total:
            return 2 ** i
    return None


def _div(a, b):
    return a / b if b else None


def analyze(prev: dict, cur: dict) -> dict:
    """Everything that happened between two :func:`snapshot` results, as
    per-request and per-second figures (``None`` where there was nothing to
    measure). The structure is what :func:`report`, the dashboard and the
    ``profile`` command show."""
    p0, p1 = prev["process"], cur["process"]
    out = {"enabled": bool(p1.get("enabled")), "pid": cur["pid"]}
    if not out["enabled"]:
        return out
    dt = max(1e-9, p1["uptime_s"] - p0["uptime_s"])
    d = _d(p0, p1)
    wall = cur["time"] - prev["time"]
    out.update(window_s=dt, cpu_cores=(cur["cpu_s"] - prev["cpu_s"]) / wall if wall >= 0.05 else None,
               rss_bytes=cur["rss_bytes"], pool=p1["pool"])

    # Encode paths and their stages.
    paths = {}
    for name, o in d["paths"].items():
        n = o["calls"]
        if n <= 0:
            continue
        stage_ms = {s: o["stage_ns"][s] / n / 1e6 for s in STAGES}
        total_ms = o["ns"] / n / 1e6
        paths[name] = {
            "encodes": n, "per_s": n / dt, "mean_ms": total_ms,
            "p50_ms": (_pct(o["latency_us_log2"], .5) or 0) / 1e3, "p99_ms": (_pct(o["latency_us_log2"], .99) or 0) / 1e3,
            "bytes": o["units"], "tokens": o["tokens"], "mb_per_s": o["units"] / dt / 1e6, "tokens_per_s": o["tokens"] / dt,
            "bytes_per_token": _div(o["units"], o["tokens"]), "mean_bytes": o["units"] / n,
            "stage_ms": stage_ms, "other_ms": max(0.0, total_ms - sum(stage_ms.values())),
            "stage_share": {s: _div(v, total_ms) for s, v in stage_ms.items()},
        }
    out["paths"] = paths
    enc_n = sum(p["encodes"] for p in paths.values())
    out["encodes"] = {"total": enc_n, "per_s": enc_n / dt,
                      "tokens_per_s": sum(p["tokens"] for p in paths.values()) / dt,
                      "mb_per_s": sum(p["bytes"] for p in paths.values()) / dt / 1e6}

    # Python API: where each call's time went.
    api = {}
    for name, o in d["api"].items():
        n = o["calls"]
        if n <= 0:
            continue
        mean = o["ns"] / n / 1e6
        rust = o["rust_ns"] / n / 1e6
        gil = o["gil_wait_ns"] / n / 1e6
        api[name] = {"calls": n, "per_s": n / dt, "units_per_s": o["units"] / dt, "mean_ms": mean,
                     "p50_ms": (_pct(o["latency_us_log2"], .5) or 0) / 1e3, "p99_ms": (_pct(o["latency_us_log2"], .99) or 0) / 1e3,
                     "rust_ms": rust, "gil_wait_ms": gil, "python_ms": max(0.0, mean - rust - gil),
                     "gil_released": o["gil_released"] / n}
    out["api"] = api

    # The BPE model's caches.
    b = d["bpe"]
    fl, fm = b["fused_lookups"], b["fused_misses"]
    model_ns = sum(o["stage_ns"]["model"] for o in d["paths"].values())
    gc, gm = b["general_calls"], b["general_merges"]
    out["bpe"] = {
        "fused": {"lookups": fl, "misses": fm, "hit_rate": _div(fl - fm, fl), "uncacheable": b["fused_uncacheable"],
                  "miss_bytes": b["fused_miss_bytes"], "merge_ms": b["fused_merge_ns"] / 1e6,
                  "merge_us_per_miss": _div(b["fused_merge_ns"] / 1e3, fm), "clears": b["fused_clears"],
                  "merge_share_of_model": _div(b["fused_merge_ns"], model_ns)},
        "general": {"calls": gc, "exact": b["general_exact"], "thread_hits": b["general_thread_hits"],
                    "shared_hits": b["general_shared_hits"], "merges": gm,
                    "hit_rate": _div(gc - gm, gc), "merge_ms": b["general_merge_ns"] / 1e6,
                    "merge_us_per_miss": _div(b["general_merge_ns"] / 1e3, gm)},
    }

    # Parallel jobs.
    jobs = {}
    for name, o in d["jobs"].items():
        if o["jobs"] > 0:
            jobs[name] = {"jobs": o["jobs"], "per_s": o["jobs"] / dt, "tasks_per_job": o["tasks"] / o["jobs"],
                          "mean_ms": o["ns"] / o["jobs"] / 1e6, "parallelism": _div(o["task_ns"], o["ns"])}
    out["jobs"] = jobs

    lw = d["lock_waits"]
    out["lock_waits"] = {k: {"per_s": v["count"] / dt, "ms_per_s": v["ns"] / dt / 1e6} for k, v in lw.items()}

    # Threads.
    prev_threads = {t["slot"]: t for t in p0["threads"]}
    threads = []
    for t in p1["threads"]:
        dt_ = _d(prev_threads.get(t["slot"], {}), t)
        busy = (dt_["task_ns"] if t["kind"] == "worker" else dt_["encode_ns"]) / dt / 1e9
        if busy < 0.001 and dt_["encodes"] <= 0 and dt_["tasks"] <= 0 and dt_["api_calls"] <= 0:
            continue
        top = max(dt_["paths"].items(), key=lambda kv: kv[1])
        threads.append({"name": t["name"] or f"slot{t['slot']}", "kind": t["kind"], "alive": t["alive"],
                        "os_tid": t["os_tid"], "busy": busy,
                        "in_api": dt_["api_ns"] / dt / 1e9 if t["kind"] != "worker" else None,
                        "encodes_per_s": dt_["encodes"] / dt, "mb_per_s": dt_["encode_bytes"] / dt / 1e6,
                        "tasks_per_s": dt_["tasks"] / dt, "main_path": top[0] if top[1] > 0 else None})
    threads.sort(key=lambda t: (t["kind"] != "worker", -t["busy"]))
    out["threads"] = threads

    # Per-tokenizer caches.
    toks = {}
    for name, tj in cur["tokenizers"].items():
        t0 = prev["tokenizers"].get(name) or {}
        entry = {"scanner_path": tj.get("scanner_path")}
        c1 = tj.get("segment_cache")
        if c1:
            c0 = t0.get("segment_cache") or {}
            dc = _d(c0, c1)
            seg = {"bytes": c1["bytes"], "approx_max_bytes": c1["approx_max_bytes"], "entries": c1["entries"],
                   "max_segment_cost": c1["max_segment_cost"]}
            if "hits" in c1:
                hb, mb = dc["hit_bytes"], dc["miss_bytes"]
                ns_per_byte = _div(dc["miss_tokenize_ns"], mb)
                sh = [_d(a, b_) for a, b_ in zip(c0.get("shards") or [{}] * len(c1["shards"]), c1["shards"])]
                acq = dc["hits"] + dc["misses"] + sum(s["inserts"] for s in sh)
                seg.update(
                    hits=dc["hits"], misses=dc["misses"],
                    hit_rate=_div(dc["hits"], dc["hits"] + dc["misses"]), byte_hit_rate=_div(hb, hb + mb),
                    miss_tokenize_ms=dc["miss_tokenize_ns"] / 1e6,
                    # What tokenizing the hit bytes would have cost at the
                    # window's miss rate per byte.
                    est_saved_ms=hb * ns_per_byte / 1e6 if ns_per_byte is not None else None,
                    rotations=sum(s["rotations"] for s in sh), evicted_bytes=sum(s["evicted_bytes"] for s in sh),
                    promotions=sum(s["promotions"] for s in sh), inserts=sum(s["inserts"] for s in sh),
                    rejected=dc["rejected_too_large"], collisions=sum(s["collisions"] for s in c1["shards"]),
                    lock_contention=_div(sum(s["lock_contended"] for s in sh), acq),
                    lock_wait_ms=sum(s["lock_wait_ns"] for s in sh) / 1e6,
                    shard_fill=[(s["young_bytes"] + s["old_bytes"]) / (2 * c1["young_budget_per_shard"]) for s in c1["shards"]],
                )
            entry["segment_cache"] = seg
        p1c = tj.get("prefix_cache")
        if p1c:
            pc = {"entries": p1c["entries"], "capacity": p1c["capacity"], "bytes": p1c["bytes"]}
            if "lookups" in p1c:
                dp = _d(t0.get("prefix_cache") or {}, p1c)
                pc.update(lookups=dp["lookups"], exact_hits=dp["exact_hits"], prefix_hits=dp["prefix_hits"],
                          misses=dp["misses"], reused_bytes=dp["reused_bytes"], reused_tokens=dp["reused_tokens"],
                          reused_share=_div(dp["reused_bytes"], dp["lookup_bytes"]))
            entry["prefix_cache"] = pc
        toks[name] = entry
    out["tokenizers"] = toks

    # The slowest encodes that ended inside the window.
    out["slowest"] = [s for s in p1.get("slowest", []) if s["at_s"] > p0["uptime_s"]]
    return out


def _fmt(v, spec=".2f", none="-"):
    return none if v is None else format(v, spec)


def report(prev: dict, cur: dict, slowest: int = 5) -> str:
    """:func:`analyze` of two snapshots, as text."""
    a = analyze(prev, cur)
    if not a["enabled"]:
        return "fastokens was built without the `observe` feature: nothing to show.\n"
    L = []
    rss = (a["rss_bytes"] or 0) / 2**20
    L.append(f"fastokens pid {a['pid']}  window {a['window_s']:.1f}s  CPU {_fmt(a['cpu_cores'])} cores  RSS {rss:.0f} MiB  "
             f"BPE pool {a['pool']['bpe_threads']} threads, global pool {a['pool']['global_threads']}")
    e = a["encodes"]
    L.append(f"encodes {e['per_s']:.1f}/s  {e['mb_per_s']:.1f} MB/s  {e['tokens_per_s'] / 1e3:.1f}k tokens/s")
    L.append("")
    L.append("where encode time goes (ms per encode; % of the encode):")
    head = f"{'path':16} {'enc/s':>8} {'mean':>8} {'p99<=':>8}  " + " ".join(f"{s[:12]:>12}" for s in STAGES) + f" {'other':>10}"
    L.append(head)
    for name, p in a["paths"].items():
        cells = []
        for s in STAGES:
            ms, share = p["stage_ms"][s], p["stage_share"][s]
            cells.append(f"{ms:7.3f} {share or 0:3.0%}" if ms > 0 else f"{'.':>12}")
        L.append(f"{name:16} {p['per_s']:8.1f} {p['mean_ms']:8.3f} {p['p99_ms']:8.3f}  " + " ".join(cells) +
                 f" {p['other_ms']:10.3f}")
    L.append("")
    L.append(f"{'API':18} {'calls/s':>9} {'mean ms':>8} {'p99<=':>8} {'rust':>8} {'GIL wait':>9} {'python':>8} {'GIL off':>8}")
    for name, o in a["api"].items():
        L.append(f"{name:18} {o['per_s']:9.1f} {o['mean_ms']:8.3f} {o['p99_ms']:8.3f} {o['rust_ms']:8.3f} "
                 f"{o['gil_wait_ms']:9.3f} {o['python_ms']:8.3f} {o['gil_released']:8.0%}")
    f, g = a["bpe"]["fused"], a["bpe"]["general"]
    L.append("")
    if f["lookups"]:
        L.append(f"BPE pretoken cache: {f['lookups'] / a['window_s']:.0f} lookups/s, hit rate {_fmt(f['hit_rate'], '.2%')}, "
                 f"{f['misses'] / a['window_s']:.0f} merges/s ({_fmt(f['merge_us_per_miss'], '.2f')} us each, "
                 f"{_fmt(f['merge_share_of_model'], '.0%')} of model time), uncacheable {f['uncacheable']}, clears {f['clears']}")
    if g["calls"]:
        L.append(f"BPE general path: {g['calls']} calls, exact {g['exact']}, thread-cache hits {g['thread_hits']}, "
                 f"shared hits {g['shared_hits']}, merges {g['merges']} ({_fmt(g['merge_us_per_miss'], '.2f')} us each)")
    if a["jobs"]:
        L.append("parallel jobs: " + ", ".join(
            f"{k} {v['per_s']:.1f}/s ({v['tasks_per_job']:.1f} tasks, {v['mean_ms']:.3f} ms, x{_fmt(v['parallelism'], '.1f')})"
            for k, v in a["jobs"].items()))
    lw = a["lock_waits"]
    if lw["read"]["per_s"] or lw["write"]["per_s"]:
        L.append(f"state lock waits: read {lw['read']['per_s']:.1f}/s ({lw['read']['ms_per_s']:.2f} ms/s), "
                 f"write {lw['write']['per_s']:.1f}/s ({lw['write']['ms_per_s']:.2f} ms/s)")
    for name, t in a["tokenizers"].items():
        s = t.get("segment_cache")
        if s:
            L.append("")
            line = f"segment cache [{name}]: {s['bytes'] / 2**20:.1f} / ~{s['approx_max_bytes'] / 2**20:.0f} MiB, {s['entries']} entries"
            if "hits" in s:
                line += (f"; hit rate {_fmt(s['hit_rate'], '.1%')} of segments, {_fmt(s['byte_hit_rate'], '.1%')} of bytes; "
                         f"saved ~{_fmt(s['est_saved_ms'], '.0f')} ms vs {s['miss_tokenize_ms']:.0f} ms tokenizing misses")
            L.append(line)
            if "hits" in s:
                L.append(f"  rotations {s['rotations']}, evicted {s['evicted_bytes'] / 2**20:.2f} MiB, promotions {s['promotions']}, "
                         f"inserts {s['inserts']}, rejected {s['rejected']}, collisions {s['collisions']}, "
                         f"lock contention {_fmt(s['lock_contention'], '.2%')} (wait {s['lock_wait_ms']:.2f} ms)")
                L.append("  shards (young+old / budget): " + " ".join(f"{x:.0%}" for x in s["shard_fill"]))
        p = t.get("prefix_cache")
        if p:
            line = f"prefix cache [{name}]: {p['entries']}/{p['capacity']} entries"
            if "lookups" in p:
                line += (f"; {p['lookups']} lookups: {p['exact_hits']} exact, {p['prefix_hits']} prefix, {p['misses']} misses; "
                         f"reused {_fmt(p['reused_share'], '.1%')} of input bytes ({p['reused_tokens']} tokens)")
            L.append(line)
    L.append("")
    L.append(f"{'thread':28} {'kind':7} {'busy':>6} {'in API':>7} {'enc/s':>8} {'tasks/s':>8}  main path")
    for t in a["threads"]:
        name = t["name"][:28] + ("" if t["alive"] else " (exited)")
        in_api = "-" if t["in_api"] is None else f"{t['in_api']:7.0%}"
        L.append(f"{name:28} {t['kind']:7} {t['busy']:6.0%} {in_api:>7} {t['encodes_per_s']:8.1f} {t['tasks_per_s']:8.1f}  {t['main_path'] or '-'}")
    if slowest and a["slowest"]:
        L.append("")
        L.append(f"slowest encodes in the window (of the {len(a['slowest'])} kept):")
        for s in a["slowest"][:slowest]:
            parts = ", ".join(f"{k} {v / 1e6:.2f}" for k, v in s["stage_ns"].items() if v)
            L.append(f"  {s['ns'] / 1e6:8.2f} ms  {s['bytes'] / 1e3:8.1f} KB  {s['tokens']:7} tokens  {s['path']:16} "
                     f"{s['thread'][:20]:20}  [{parts}]")
    return "\n".join(L) + "\n"


class Profile:
    """A profiled window: :func:`snapshot` before and after, and their
    :func:`analyze` as ``data``."""

    def __init__(self, tokenizers=None, reset: bool = True):
        self._tokenizers = _named(tokenizers)
        self._reset = reset
        self.before = self.after = None
        self.data: dict | None = None

    def __enter__(self):
        if self._reset:
            reset_slow()
        self.before = snapshot(self._tokenizers)
        return self

    def __exit__(self, *exc):
        self.after = snapshot(self._tokenizers)
        self.data = analyze(self.before, self.after)
        return False

    def report(self, slowest: int = 5) -> str:
        if self.after is None:
            raise RuntimeError("the profile has not ended yet")
        return report(self.before, self.after, slowest)


def profile(tokenizers=None, reset: bool = True) -> Profile:
    """Profile the ``with`` block: ``with profile({"glm": tok}) as p: ...``,
    then ``p.report()`` or ``p.data``. ``reset`` forgets the slowest encodes
    kept before the block, so the ones listed ran inside it."""
    return Profile(tokenizers, reset)


# ── Prometheus text format ────────────────────────────────────────────────────


def _esc(v) -> str:
    return str(v).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def prometheus(snap: dict) -> str:
    """``snap`` (from :func:`snapshot`) in the Prometheus text format."""
    out: list[str] = []
    seen: set[str] = set()

    def m(name, value, labels=None, kind="counter", help_=""):
        if value is None:
            return
        family = name
        if kind == "histogram":
            for suffix in ("_bucket", "_sum", "_count"):
                if family.endswith(suffix):
                    family = family[: -len(suffix)]
                    break
        if family not in seen:
            seen.add(family)
            out.append(f"# HELP {family} {help_ or family}")
            out.append(f"# TYPE {family} {kind}")
        lab = ""
        if labels:
            lab = "{" + ",".join(f'{k}="{_esc(v)}"' for k, v in labels.items()) + "}"
        out.append(f"{name}{lab} {value}")

    def hist(name, buckets, labels, scale, total_s, help_):
        acc = 0
        for i, c in enumerate(buckets):
            acc += c
            m(f"{name}_bucket", acc, {**labels, "le": f"{(2 ** i) * scale:g}"}, "histogram", help_)
        m(f"{name}_bucket", acc, {**labels, "le": "+Inf"}, "histogram", help_)
        m(f"{name}_sum", total_s, labels, "histogram", help_)
        m(f"{name}_count", acc, labels, "histogram", help_)

    m("fastokens_process_cpu_seconds_total", snap["cpu_s"], help_="CPU time of the process")
    m("fastokens_process_rss_bytes", snap["rss_bytes"], kind="gauge", help_="resident memory")
    p = snap["process"]
    m("fastokens_observe_enabled", int(bool(p.get("enabled"))), kind="gauge", help_="built with the observe feature")
    if p.get("enabled"):
        m("fastokens_pool_threads", p["pool"]["bpe_threads"], {"pool": "bpe"}, "gauge", "threads in a pool")
        m("fastokens_pool_threads", p["pool"]["global_threads"], {"pool": "global"}, "gauge")
        for kind, v in p["jobs"].items():
            lab = {"kind": kind}
            m("fastokens_jobs_total", v["jobs"], lab, help_="parallel jobs by what started them")
            m("fastokens_job_tasks_total", v["tasks"], lab, help_="tasks of parallel jobs")
            m("fastokens_job_seconds_total", v["ns"] / 1e9, lab, help_="wall time of parallel jobs")
            m("fastokens_job_task_seconds_total", v["task_ns"] / 1e9, lab, help_="summed time of their tasks")
        for path, o in p["paths"].items():
            lab = {"path": path}
            m("fastokens_encode_total", o["calls"], lab, help_="encodes by pipeline branch")
            m("fastokens_encode_bytes_total", o["units"], lab, help_="input bytes encoded")
            m("fastokens_encode_tokens_total", o["tokens"], lab, help_="tokens produced")
            for stage, ns in o["stage_ns"].items():
                m("fastokens_encode_stage_seconds_total", ns / 1e9, {**lab, "stage": stage}, help_="encode time by stage")
            hist("fastokens_encode_latency_seconds", o["latency_us_log2"], lab, 1e-6, o["ns"] / 1e9, "encode latency")
        for api, o in p["api"].items():
            lab = {"api": api}
            m("fastokens_api_calls_total", o["calls"], lab, help_="Python API calls")
            m("fastokens_api_units_total", o["units"], lab, help_="input bytes (ids for decode and lists)")
            m("fastokens_api_rust_seconds_total", o["rust_ns"] / 1e9, lab, help_="time in Rust, state lock included")
            m("fastokens_api_gil_released_total", o["gil_released"], lab, help_="calls that released the GIL")
            m("fastokens_api_gil_wait_seconds_total", o["gil_wait_ns"] / 1e9, lab,
              help_="time from the end of GIL-free Rust work to holding the GIL again")
            hist("fastokens_api_latency_seconds", o["latency_us_log2"], lab, 1e-6, o["ns"] / 1e9, "API call latency")
        for key, v in p["bpe"].items():
            if key.endswith("_ns"):
                m(f"fastokens_bpe_{key[:-3]}_seconds_total", v / 1e9, help_=f"BPE {key[:-3]} time")
            else:
                m(f"fastokens_bpe_{key}_total", v, help_=f"BPE cache {key}")
        for mode, w in p["lock_waits"].items():
            m("fastokens_state_lock_waits_total", w["count"], {"mode": mode}, help_="blocked state-lock acquisitions")
            m("fastokens_state_lock_wait_seconds_total", w["ns"] / 1e9, {"mode": mode}, help_="time blocked on the state lock")
        for t in p["threads"]:
            lab = {"thread": t["name"] or f"slot{t['slot']}", "kind": t["kind"], "slot": t["slot"]}
            m("fastokens_thread_encode_seconds_total", t["encode_ns"] / 1e9, lab, help_="encode time on a thread")
            m("fastokens_thread_task_seconds_total", t["task_ns"] / 1e9, lab, help_="parallel-task time on a thread")
            m("fastokens_thread_api_seconds_total", t["api_ns"] / 1e9, lab, help_="Python API time on a thread")
            m("fastokens_thread_alive", int(t["alive"]), lab, "gauge", "whether the thread is running")
    for name, tj in snap["tokenizers"].items():
        lab = {"tokenizer": name}
        c = tj.get("segment_cache")
        if c:
            for key in ("entries", "bytes", "approx_max_bytes"):
                m(f"fastokens_segment_cache_{key}", c.get(key), lab, "gauge", f"segment cache {key}")
            for key in ("hits", "misses", "hit_bytes", "miss_bytes", "rejected_too_large", "rejected_bytes"):
                m(f"fastokens_segment_cache_{key}_total", c.get(key), lab, help_=f"segment cache {key}")
            if "miss_tokenize_ns" in c:
                m("fastokens_segment_cache_miss_tokenize_seconds_total", c["miss_tokenize_ns"] / 1e9, lab,
                  help_="time spent tokenizing missed segments")
            for i, sh in enumerate(c.get("shards", [])):
                sl = {**lab, "shard": i}
                for gen in ("young", "old"):
                    m("fastokens_segment_cache_shard_bytes", sh[f"{gen}_bytes"], {**sl, "gen": gen}, "gauge", "shard bytes")
                    m("fastokens_segment_cache_shard_entries", sh[f"{gen}_entries"], {**sl, "gen": gen}, "gauge", "shard entries")
                for key in ("rotations", "evicted_entries", "evicted_bytes", "promotions", "inserts",
                            "replaced", "collisions", "lock_contended"):
                    m(f"fastokens_segment_cache_shard_{key}_total", sh[key], sl, help_=f"shard {key}")
                m("fastokens_segment_cache_shard_lock_wait_seconds_total", sh["lock_wait_ns"] / 1e9, sl,
                  help_="time waiting for a shard lock")
        pc = tj.get("prefix_cache")
        if pc:
            m("fastokens_prefix_cache_entries", pc["entries"], lab, "gauge", "prefix cache entries")
            for key in ("lookups", "exact_hits", "prefix_hits", "misses", "lookup_bytes", "reused_bytes", "reused_tokens"):
                m(f"fastokens_prefix_cache_{key}_total", pc.get(key), lab, help_=f"prefix cache {key}")
    return "\n".join(out) + "\n"


# ── HTTP server ───────────────────────────────────────────────────────────────


class _Server(ThreadingHTTPServer):
    # The default listen backlog of 5 resets connections when a scraper and a
    # few dashboards poll at once.
    request_queue_size = 128
    daemon_threads = True


def serve(tokenizers=None, port: int = 9464, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Serve the dashboard (``/``), :func:`snapshot` (``/json``) and
    :func:`prometheus` (``/metrics``) from a daemon thread of this process;
    ``POST /reset-slow`` calls :func:`reset_slow`, ``POST /analyze`` with a
    previous snapshot as the body returns the current one and their
    :func:`analyze`. Binds to localhost by
    default (``port=0`` picks a free port, see ``server.server_address``);
    ``server.shutdown()`` stops it."""
    from ._observe_html import DASHBOARD

    # Resolve now, so a wrong argument fails here rather than on every request.
    natives = _named(tokenizers)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, ctype: str):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path == "/json":
                    body, ctype = json.dumps(snapshot(natives)).encode(), "application/json"
                elif path == "/metrics":
                    body, ctype = prometheus(snapshot(natives)).encode(), "text/plain; version=0.0.4; charset=utf-8"
                elif path in ("/", "/index.html"):
                    body, ctype = DASHBOARD.encode(), "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
            except Exception as e:  # never take the server thread down
                self.send_error(500, explain=repr(e))
                return
            self._send(body, ctype)

        def do_POST(self):
            path = self.path.split("?")[0]
            try:
                if path == "/reset-slow":
                    reset_slow()
                    self._send(b"{}", "application/json")
                elif path == "/analyze":
                    # The dashboard sends its previous snapshot and gets the
                    # current one with analyze() of the window between, so
                    # it shows exactly what report() would.
                    n = int(self.headers.get("Content-Length") or 0)
                    prev = json.loads(self.rfile.read(n) or b"null") if n else None
                    cur = snapshot(natives)
                    body = {"snapshot": cur, "analysis": analyze(prev, cur) if prev else None}
                    self._send(json.dumps(body).encode(), "application/json")
                else:
                    self.send_error(404)
            except Exception as e:
                self.send_error(500, explain=repr(e))

        def log_message(self, *args):
            pass

    server = _Server((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="fastokens-observe", daemon=True).start()
    return server


# ── command line ──────────────────────────────────────────────────────────────


def _fetch(url):
    with urllib.request.urlopen(url.rstrip("/") + "/json", timeout=5) as r:
        return json.loads(r.read())


def _reset_remote(url):
    req = urllib.request.Request(url.rstrip("/") + "/reset-slow", data=b"", method="POST")
    with urllib.request.urlopen(req, timeout=5):
        pass


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fastokens.observe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    top = sub.add_parser("top", help="live text view of a process running observe.serve()")
    top.add_argument("--url", default="http://127.0.0.1:9464")
    top.add_argument("--interval", type=float, default=2.0)
    top.add_argument("--once", action="store_true", help="print one window and exit")
    prof = sub.add_parser("profile", help="profile a running process for a while and print the analysis")
    prof.add_argument("--url", default="http://127.0.0.1:9464")
    prof.add_argument("--seconds", type=float, default=10.0)
    prof.add_argument("--slowest", type=int, default=10, help="slowest encodes to list")
    prof.add_argument("--keep-slow", action="store_true", help="do not reset the slowest-encodes list first")
    prof.add_argument("--json", action="store_true", help="print analyze() as JSON instead of text")
    dump = sub.add_parser("dump", help="print one /json snapshot")
    dump.add_argument("--url", default="http://127.0.0.1:9464")
    a = ap.parse_args(argv)
    if a.cmd == "dump":
        print(json.dumps(_fetch(a.url), indent=1))
        return 0
    if a.cmd == "profile":
        if not a.keep_slow:
            _reset_remote(a.url)
        before = _fetch(a.url)
        time.sleep(a.seconds)
        after = _fetch(a.url)
        if a.json:
            print(json.dumps(analyze(before, after), indent=1))
        else:
            sys.stdout.write(report(before, after, a.slowest))
        return 0
    prev = _fetch(a.url)
    while True:
        time.sleep(a.interval)
        cur = _fetch(a.url)
        text = report(prev, cur)
        if a.once:
            sys.stdout.write(text)
            return 0
        sys.stdout.write("\x1b[2J\x1b[H" + text)
        sys.stdout.flush()
        prev = cur


if __name__ == "__main__":
    sys.exit(main())
