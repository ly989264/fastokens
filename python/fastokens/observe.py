"""Look inside a running fastokens: its thread pools, the threads doing the
work, which encode path inputs take, API latency and GIL waits, and each
tokenizer's segment cache shard by shard.

Needs a build with the ``observe`` cargo feature; without it every counter is
absent and :func:`enabled` is ``False``::

    maturin build --release --features observe

In the process that tokenizes (a server, a benchmark)::

    import fastokens.observe as obs
    obs.serve({"glm": tok}, port=9464)   # dashboard at http://127.0.0.1:9464
                                         # plus /json and /metrics (Prometheus)
    obs.snapshot({"glm": tok})           # the same data as a dict

``tok`` may be a ``fastokens._native.Tokenizer``, the shim that
``fastokens.patch_transformers()`` installs, or a ``transformers`` fast
tokenizer wrapping either. From another terminal::

    python -m fastokens.observe top --url http://127.0.0.1:9464

Counters only grow; every rate shown is the difference of two snapshots over
their ``uptime_s`` difference. Histograms are log2 buckets: bucket ``i``
counts values in ``[2**(i-1), 2**i)`` (microseconds or bytes), bucket 0 zeros.
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

__all__ = ["enabled", "snapshot", "serve", "prometheus", "report"]


def enabled() -> bool:
    """Whether this build has the ``observe`` feature."""
    return _native.observe_enabled()


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
    if tokenizers is not None and not isinstance(tokenizers, dict):
        tokenizers = {"tokenizer": tokenizers}
    natives = {name: _native_of(t) for name, t in (tokenizers or {}).items()}
    return {
        "time": time.time(),
        "pid": os.getpid(),
        "cpu_s": _cpu_s(),
        "rss_bytes": _rss_bytes(),
        "process": json.loads(_native.observe_snapshot()),
        "tokenizers": {name: json.loads(t.observe()) for name, t in natives.items()},
    }


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
        # log2 buckets -> cumulative Prometheus buckets with le = 2**i * scale
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
            m("fastokens_jobs_total", v["jobs"], {"kind": kind}, help_="parallel jobs by what started them")
            m("fastokens_job_tasks_total", v["tasks"], {"kind": kind}, help_="tasks of parallel jobs")
            m("fastokens_job_seconds_total", v["ns"] / 1e9, {"kind": kind}, help_="wall time of parallel jobs")
        for path, o in p["paths"].items():
            lab = {"path": path}
            m("fastokens_encode_total", o["calls"], lab, help_="encodes by pipeline branch")
            m("fastokens_encode_bytes_total", o["units"], lab, help_="input bytes encoded")
            hist("fastokens_encode_latency_seconds", o["latency_us_log2"], lab, 1e-6, o["ns"] / 1e9, "encode latency")
        for api, o in p["api"].items():
            lab = {"api": api}
            m("fastokens_api_calls_total", o["calls"], lab, help_="Python API calls")
            m("fastokens_api_units_total", o["units"], lab, help_="input bytes (ids for decode)")
            m("fastokens_api_gil_released_total", o["gil_released"], lab, help_="calls that released the GIL")
            m("fastokens_api_gil_wait_seconds_total", o["gil_wait_ns"] / 1e9, lab,
              help_="time from the end of GIL-free Rust work to holding the GIL again")
            hist("fastokens_api_latency_seconds", o["latency_us_log2"], lab, 1e-6, o["ns"] / 1e9, "API call latency")
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
        c = tj.get("segment_cache")
        if not c:
            continue
        lab = {"tokenizer": name}
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
    return "\n".join(out) + "\n"


# ── HTTP server ───────────────────────────────────────────────────────────────


class _Server(ThreadingHTTPServer):
    # The default listen backlog of 5 resets connections when a scraper and a
    # few dashboards poll at once.
    request_queue_size = 128
    daemon_threads = True


def serve(tokenizers=None, port: int = 9464, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Serve the dashboard (``/``), :func:`snapshot` (``/json``) and
    :func:`prometheus` (``/metrics``) from a daemon thread of this process.
    Binds to localhost by default (``port=0`` picks a free port, see
    ``server.server_address``); ``server.shutdown()`` stops it."""
    from ._observe_html import DASHBOARD

    # Resolve now, so a wrong argument fails here rather than on every request.
    if tokenizers is not None and not isinstance(tokenizers, dict):
        tokenizers = {"tokenizer": tokenizers}
    tokenizers = {name: _native_of(t) for name, t in (tokenizers or {}).items()}

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?")[0]
            try:
                if path == "/json":
                    body, ctype = json.dumps(snapshot(tokenizers)).encode(), "application/json"
                elif path == "/metrics":
                    body, ctype = prometheus(snapshot(tokenizers)).encode(), "text/plain; version=0.0.4; charset=utf-8"
                elif path in ("/", "/index.html"):
                    body, ctype = DASHBOARD.encode(), "text/html; charset=utf-8"
                else:
                    self.send_error(404)
                    return
            except Exception as e:  # never take the server thread down
                self.send_error(500, explain=repr(e))
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = _Server((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="fastokens-observe", daemon=True).start()
    return server


# ── text report (python -m fastokens.observe top) ─────────────────────────────


def _pct(hist, q):
    """Upper bound of the log2 bucket holding quantile ``q``."""
    total = sum(hist)
    if not total:
        return None
    acc = 0
    for i, c in enumerate(hist):
        acc += c
        if acc >= q * total:
            return 2 ** i
    return None


def _d(a, b):
    """b - a for numbers and lists, recursively for dicts."""
    if isinstance(b, dict):
        return {k: _d(a.get(k) if isinstance(a, dict) else None, v) for k, v in b.items()}
    if isinstance(b, list) and b and all(isinstance(x, (int, float)) for x in b):
        a = a if isinstance(a, list) and len(a) == len(b) else [0] * len(b)
        return [y - x for x, y in zip(a, b)]
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        return b - (a if isinstance(a, (int, float)) else 0)
    return b


def report(prev: dict, cur: dict) -> str:
    """What happened between two :func:`snapshot` results, as text."""
    p0, p1 = prev["process"], cur["process"]
    lines = []
    if not p1.get("enabled"):
        return "fastokens was built without the `observe` feature: nothing to show.\n"
    dt = max(1e-9, p1["uptime_s"] - p0["uptime_s"])
    d = _d(p0, p1)
    cpu = (cur["cpu_s"] - prev["cpu_s"]) / dt
    rss = (cur["rss_bytes"] or 0) / 2**20
    lines.append(f"fastokens pid {cur['pid']}  window {dt:.1f}s  CPU {cpu:.2f} cores  RSS {rss:.0f} MiB  "
                 f"BPE pool {p1['pool']['bpe_threads']} threads, global pool {p1['pool']['global_threads']}")
    lines.append("")
    lines.append(f"{'API':18} {'calls/s':>9} {'MB/s':>8} {'mean ms':>8} {'p50<=':>8} {'p99<=':>8} {'GIL off':>8} {'GIL wait':>10}")
    for api, o in d["api"].items():
        if o["calls"] <= 0:
            continue
        p50, p99 = _pct(o["latency_us_log2"], .5), _pct(o["latency_us_log2"], .99)
        lines.append(f"{api:18} {o['calls'] / dt:9.1f} {o['units'] / dt / 1e6:8.1f} {o['ns'] / o['calls'] / 1e6:8.3f} "
                     f"{(p50 or 0) / 1e3:8.3f} {(p99 or 0) / 1e3:8.3f} {o['gil_released'] / o['calls']:8.0%} "
                     f"{o['gil_wait_ns'] / o['calls'] / 1e6:7.3f} ms")
    lines.append("")
    lines.append(f"{'encode path':18} {'runs/s':>9} {'MB/s':>8} {'mean ms':>8} {'p99<=':>8}")
    for path, o in d["paths"].items():
        if o["calls"] > 0:
            p99 = _pct(o["latency_us_log2"], .99)
            lines.append(f"{path:18} {o['calls'] / dt:9.1f} {o['units'] / dt / 1e6:8.1f} "
                         f"{o['ns'] / o['calls'] / 1e6:8.3f} {(p99 or 0) / 1e3:8.3f}")
    kinds = [f"{k} {v['jobs'] / dt:.1f}/s ({v['tasks'] / v['jobs']:.1f} tasks, {v['ns'] / v['jobs'] / 1e3:.0f} us)"
             for k, v in d["jobs"].items() if v["jobs"] > 0]
    lines.append("")
    lines.append("parallel jobs: " + (", ".join(kinds) if kinds else "none"))
    lw = d["lock_waits"]
    lines.append(f"state lock: read waits/s {lw['read']['count'] / dt:.1f} ({lw['read']['ns'] / dt / 1e6:.2f} ms/s)  "
                 f"write waits/s {lw['write']['count'] / dt:.1f} ({lw['write']['ns'] / dt / 1e6:.2f} ms/s)")
    lines.append("")
    lines.append(f"{'thread':28} {'kind':7} {'busy':>6} {'in API':>7} {'enc/s':>8} {'tasks/s':>8}")
    prev_threads = {t["slot"]: t for t in p0["threads"]}
    rows = []
    for t in p1["threads"]:
        dt_ = _d(prev_threads.get(t["slot"], {}), t)
        busy = (dt_["task_ns"] if t["kind"] == "worker" else dt_["encode_ns"]) / dt / 1e9
        if busy < 0.001 and dt_["encodes"] <= 0 and dt_["tasks"] <= 0 and dt_["api_calls"] <= 0:
            continue
        rows.append((t["kind"] != "worker", -busy, t, dt_, busy))
    for _, _, t, dt_, busy in sorted(rows, key=lambda r: r[:2]):
        name = (t["name"] or f"slot{t['slot']}")[:28] + ("" if t["alive"] else " (exited)")
        api = "-" if t["kind"] == "worker" else f"{dt_['api_ns'] / dt / 1e9:7.0%}"
        lines.append(f"{name:28} {t['kind']:7} {busy:6.0%} {api:>7} {dt_['encodes'] / dt:8.1f} {dt_['tasks'] / dt:8.1f}")
    for name, tj in cur["tokenizers"].items():
        c1 = tj.get("segment_cache")
        if not c1:
            continue
        c0 = (prev["tokenizers"].get(name) or {}).get("segment_cache") or {}
        dc = _d(c0, c1)
        lines.append("")
        head = (f"segment cache [{name}]: {c1['bytes'] / 2**20:.1f} / ~{c1['approx_max_bytes'] / 2**20:.0f} MiB, "
                f"{c1['entries']} entries")
        if "hits" in c1:
            hb, mb = dc["hit_bytes"], dc["miss_bytes"]
            head += (f"; window hit rate {dc['hits'] / max(1, dc['hits'] + dc['misses']):.1%} of segments, "
                     f"{hb / max(1, hb + mb):.1%} of bytes")
        lines.append(head)
        if "shards" in c1:
            prev_sh = c0.get("shards") or [{}] * len(c1["shards"])
            sh = [_d(a, b) for a, b in zip(prev_sh, c1["shards"])]
            acq = dc["hits"] + dc["misses"] + sum(s["inserts"] for s in sh)
            cont = sum(s["lock_contended"] for s in sh)
            lines.append(f"  rotations/s {sum(s['rotations'] for s in sh) / dt:.2f}  evicted {sum(s['evicted_bytes'] for s in sh) / dt / 2**20:.2f} MiB/s  "
                         f"promotions/s {sum(s['promotions'] for s in sh) / dt:.1f}  inserts/s {sum(s['inserts'] for s in sh) / dt:.1f}  "
                         f"rejected/s {dc['rejected_too_large'] / dt:.2f}  collisions {sum(s['collisions'] for s in c1['shards'])}")
            lines.append(f"  lock contended {cont / max(1, acq):.2%} of {acq / dt:.0f} acquisitions/s, "
                         f"wait {sum(s['lock_wait_ns'] for s in sh) / dt / 1e6:.2f} ms/s; misses tokenizing "
                         f"{dc['miss_tokenize_ns'] / dt / 1e6:.1f} ms/s")
            per = 2 * c1["young_budget_per_shard"]
            lines.append("  shards (young+old / budget): " + " ".join(
                f"{(s['young_bytes'] + s['old_bytes']) / per:.0%}" for s in c1["shards"]))
    return "\n".join(lines) + "\n"


def _fetch(url):
    with urllib.request.urlopen(url.rstrip("/") + "/json", timeout=5) as r:
        return json.loads(r.read())


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m fastokens.observe")
    sub = ap.add_subparsers(dest="cmd", required=True)
    top = sub.add_parser("top", help="live text view of a process running observe.serve()")
    top.add_argument("--url", default="http://127.0.0.1:9464")
    top.add_argument("--interval", type=float, default=2.0)
    top.add_argument("--once", action="store_true", help="print one window and exit")
    dump = sub.add_parser("dump", help="print one /json snapshot")
    dump.add_argument("--url", default="http://127.0.0.1:9464")
    a = ap.parse_args(argv)
    if a.cmd == "dump":
        print(json.dumps(_fetch(a.url), indent=1))
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
