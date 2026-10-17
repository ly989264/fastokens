"""CPU sampling of a tokenizing process, and the split of a call into the
layers around fastokens. Used by ``python -m fastokens.observe sample`` and
``python -m fastokens.observe layers``; see docs/OBSERVE.zh-CN.md.

Sampling uses what the OS ships: macOS ``/usr/bin/sample``, Linux ``perf``
(the Linux path is not verified). Every sample is a stack; the frame on top
(the self time) decides its category, and frames that belong to no component
of their own (malloc, memcpy, hashing, ...) are also charged to the pipeline
stage (fastokens' or HF tokenizers') that called them, so "allocation inside pre-tokenization" is visible.
Rust symbols are read without a demangler: their path segments are
length-prefixed (``10tokenizers6models3bpe``), which is enough to recognise
modules.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict

# ── parsing ──────────────────────────────────────────────────────────────────

_SAMPLE_THREAD = re.compile(r"^ {4}(\d+) (Thread_\d+\S*)\s*(.*)$")
_SAMPLE_FRAME = re.compile(r"^([ +!:|]*?)(\d+) (.+?)  \(in (.+?)\)")
_SAMPLE_FRAME_NOLIB = re.compile(r"^([ +!:|]*?)(\d+) (\S.*?)\s*$")


def parse_macos_sample(text: str) -> list[tuple[str, list[tuple[str, str]], int]]:
    """``(thread, frames root..leaf, self samples)`` from the call graph of a
    macOS ``sample`` report. A node's self samples are its count minus its
    children's."""
    if "Call graph:" not in text:
        raise ValueError("not a macOS `sample` report (no 'Call graph:' section)")
    body = text.split("Call graph:", 1)[1]
    for end in ("\nTotal number in stack", "\nSort by top of stack", "\nBinary Images:"):
        body = body.split(end, 1)[0]
    out = []
    # stack of [prefix_len, frame, count, children_count]
    stack: list[list] = []
    thread = None

    def flush_to(depth):
        while stack and stack[-1][0] >= depth:
            node = stack.pop()
            self_n = node[2] - node[3]
            if self_n > 0:
                out.append((thread, [n[1] for n in stack] + [node[1]], self_n))

    for line in body.splitlines():
        if not line.strip():
            continue
        m = _SAMPLE_THREAD.match(line)
        if m:
            flush_to(0)
            name = m.group(3).strip()
            thread = f"{m.group(2)} {name}".strip()
            stack.append([4, ("<thread>", ""), int(m.group(1)), 0])
            continue
        m = _SAMPLE_FRAME.match(line)
        if m:
            prefix, n, sym, lib = len(m.group(1)), int(m.group(2)), m.group(3).strip(), m.group(4)
        else:
            m = _SAMPLE_FRAME_NOLIB.match(line)
            if not m or thread is None:
                continue
            prefix, n, sym, lib = len(m.group(1)), int(m.group(2)), m.group(3).split("  ")[0].strip(), ""
        flush_to(prefix)
        if stack:
            stack[-1][3] += n
        stack.append([prefix, (sym, lib), n, 0])
    flush_to(0)
    # drop the synthetic thread frame
    return [(t, [f for f in frames if f[0] != "<thread>"], n) for t, frames, n in out]


_PERF_HEAD = re.compile(r"^(\S.*?)\s+(\d+)(?:/(\d+))?\s+")
_PERF_FRAME = re.compile(r"^\s+[0-9a-fA-F]+\s+(.+?)(?:\+0x[0-9a-fA-F]+)?\s+\((.*)\)\s*$")


def parse_perf_script(text: str) -> list[tuple[str, list[tuple[str, str]], int]]:
    """Samples from ``perf script`` output (one sample = one stack, leaf
    first in the input). Not verified on Linux."""
    out = []
    thread, frames = None, []

    def done():
        if thread is not None and frames:
            out.append((thread, list(reversed(frames)), 1))

    for line in text.splitlines():
        if not line.strip():
            done()
            thread, frames = None, []
            continue
        if not line[0].isspace():
            done()
            m = _PERF_HEAD.match(line)
            thread = f"{m.group(1)}/{m.group(3) or m.group(2)}" if m else line.split()[0]
            frames = []
            continue
        m = _PERF_FRAME.match(line)
        if m:
            frames.append((m.group(1), os.path.basename(m.group(2))))
    done()
    return out


def parse_any(text: str):
    return parse_macos_sample(text) if "Call graph:" in text else parse_perf_script(text)


# ── symbols ──────────────────────────────────────────────────────────────────

def idents(sym: str) -> list[str]:
    """Path segments of a mangled Rust symbol (v0 ``_R`` or legacy ``_ZN``):
    length-prefixed identifiers followed by a tag, a digit or the end. A C or
    demangled name is split on ``::`` instead."""
    if not (sym.startswith("_R") or sym.startswith("_ZN") or sym.startswith("__ZN")):
        # demangled (possibly with legacy escapes: _$LT$core..hash..sip..Hasher$GT$)
        for a, b in (("$LT$", "<"), ("$GT$", ">"), ("$u20$", " "), ("$C$", ","), ("$RF$", "&"), ("..", "::")):
            sym = sym.replace(a, b)
        return [p for p in re.split(r"::|<|>| |,|&|\(|\)|\[|\]", sym) if p and p != "_"]
    v0 = sym.startswith("_R")
    out, i, n = [], 0, len(sym)
    while i < n:
        c = sym[i]
        if c.isdigit() and (i == 0 or not sym[i - 1].isdigit()):
            j = i
            while j < n and sym[j].isdigit():
                j += 1
            k = int(sym[i:j])
            start = j + 1 if v0 and j < n and sym[j] == "_" and k and j + 1 < n and (sym[j + 1].isdigit() or sym[j + 1] == "_") else j
            ident = sym[start:start + k]
            after = sym[start + k:start + k + 1]
            # v0 writes an identifier that starts with "_" or a digit after an
            # explicit "_" separator, so without one it must start with a letter
            # (this also skips disambiguators such as "s3_")
            ok_start = ident[:1].isalpha() or (ident[:1] == "_" and (start > j or not v0))
            if 0 < k <= 64 and len(ident) == k and ok_start \
                    and (after == "" or after.isdigit() or after.isupper()):
                out.append(ident)
                i = start + k
                continue
        i += 1
    return [x for x in out if not re.fullmatch(r"h[0-9a-f]{16}", x)]


def short_name(sym: str) -> str:
    parts = idents(sym)
    if sym.startswith(("_R", "_ZN", "__ZN")) and parts:
        return "::".join(parts[-4:])
    return sym


# ── classification ───────────────────────────────────────────────────────────

# Categories, in report order.
CATEGORIES = [
    "stage: special-token matching", "stage: normalize", "stage: pre-tokenize",
    "stage: BPE", "stage: segment cache", "stage: prefix cache", "stage: post-process",
    "stage: decode", "Rust: other (fastokens)", "binding: input str -> UTF-8",
    "binding: building Python lists / objects", "binding: Rust side (pyo3, conversions)",
    "Rust: other (HF tokenizers)",
    "allocator (malloc/free/realloc)", "memcpy/memmove/memset", "hashing / hash tables",
    "regex engine", "locks / condvars", "thread pool: scheduling / spin",
    "Python: freeing objects (lists, Encodings)", "allocator (Python obmalloc)",
    "Python interpreter (caller code, transformers, Jinja)",
    "GIL wait", "thread pool: blocked", "idle / blocked (other)", "other",
]
BLOCKED = {"GIL wait", "thread pool: blocked", "idle / blocked (other)"}

_BLOCKING_LEAF = re.compile(
    r"__psynch_cvwait|__psynch_mutexwait|__semwait_signal|semaphore_.*wait|mach_msg|__workq_kernreturn|"
    r"kevent|__select|^poll$|__ulock_wait|^__wait4|^read$|^__read_nocancel|recvfrom|^__accept|"
    r"futex|epoll_wait|nanosleep|^__sleep|pthread_cond_wait|sem_wait|sigtimedwait|^syscall$")
_SPIN_LEAF = re.compile(r"sched_yield|swtch_pri|thread_switch|spin_loop|__yield")
_ALLOC = re.compile(r"(^|_)(malloc|calloc|realloc|free|posix_memalign|aligned_alloc)($|_)|^_?malloc_|"
                    r"^_?xzm_|_rdl_alloc|__rust_alloc|__rust_dealloc|"
                    r"__rust_realloc|^_?rjem_|^_?mi_|^_?je_")
_MEMCPY = re.compile(r"memmove|memcpy|memset|bzero|_platform_mem")
_PY_ALLOC = re.compile(r"^_?PyObject_(Malloc|Calloc|Realloc|Free)$|^pymalloc_|^_?PyMem_|^allocate_from_new_pool$|"
                       r"^_PyObject_GC_(New|Alloc|Link|Del)|^gc_alloc")
_PY_FREE = re.compile(r"dealloc$|^_Py_Dealloc$|^tupledealloc$|^subtype_dealloc$|^_PyObject_GC_Del$")
_PY_UTF8 = re.compile(r"utf8_encoder|unicode_encode_utf8|PyUnicode_AsUTF8|PyUnicode_EncodeUTF8")
_PY_API = re.compile(r"^_?Py[A-Z_]")
_PY_LIB = re.compile(r"^(Python|libpython[\d.]*\.(dylib|so.*))$")
_HASH_IDS = {"hash", "hasher", "siphasher", "sip", "ahash", "fxhash", "hashbrown", "hash_bytes",
             "hash_one", "make_hasher", "RandomState", "BuildHasher", "RawTable", "fnv"}
_REGEX_IDS = {"pcre2", "fancy_regex", "regex_automata", "regex", "onig", "regex_syntax"}
_REGEX_C = re.compile(r"^_?pcre2_|^onig|^match_at$|^search_in_range$|^jit_machine_stack_exec$|^_?pcre2_jit")
_LOCK = re.compile(r"pthread_mutex|__psynch_cvsignal|__psynch_cvbroad|__psynch_mutexdrop|os_unfair_lock|"
                   r"_pthread_cond_signal|pthread_cond_broadcast|pthread_cond_signal|parking_lot")
_POOL_IDS = {"rayon_core", "rayon", "crossbeam_deque", "crossbeam_epoch", "crossbeam_utils", "fanout"}

# Module / type names that place a frame in a pipeline stage. Container types
# used by every stage (HF's tokenizer::normalizer::NormalizedString and
# tokenizer::pre_tokenizer::PreTokenizedString, fastokens' pre_tokenized) are
# deliberately absent: such frames take the stage of their caller.
_STAGE_IDS = [
    ("stage: segment cache", {"segment_cache", "SegmentCache"}),
    ("stage: prefix cache", {"prefix_cache", "InputCache"}),
    ("stage: special-token matching", {"added_tokens", "added_vocabulary", "AddedVocabulary", "aho_corasick",
                                           "daachorse"}),
    ("stage: normalize", {"normalizers", "unicode_normalization", "unicode_normalization_alignments",
                              "icu_normalizer"}),
    ("stage: BPE", {"bpe", "BPE"}),
    ("stage: pre-tokenize", {"pre_tokenizers", "split", "scan", "byte_level", "unicode_class", "ByteLevel"}),
    ("stage: post-process", {"processors", "post_process", "post_processor"}),
    ("stage: decode", {"decoders", "DecodeStream"}),
]


def _lib_kind(lib: str) -> str:
    lib = os.path.basename(lib or "")
    if lib.startswith("_native"):
        return "fastokens"
    if lib.startswith("tokenizers"):
        return "hf"
    if _PY_LIB.match(lib) or re.search(r"\.cpython-\d+.*\.so$|^_.*\.(so|pyd)$", lib):
        return "python"
    if lib.startswith(("libsystem_", "libc.", "libc-", "libpthread", "libdyld", "ld-linux", "libm.")):
        return "system"
    return "other"


def _frame_kind(frame) -> str:
    """fastokens / hf / python / system / other: by library, or for a
    statically linked binary by the crate in the symbol."""
    sym, lib = frame
    kind = _lib_kind(lib)
    if kind == "other" and sym.startswith(("_R", "_ZN", "__ZN")):
        ids = set(idents(sym))
        if "fastokens" in ids:
            return "fastokens"
        if "tokenizers" in ids:
            return "hf"
    return kind


def _stage_of(frame) -> str | None:
    """The fastokens / HF pipeline stage a Rust frame belongs to, if any. A
    frame whose name mentions several stages (a generic over all of them, like
    TokenizerImpl<ModelWrapper, NormalizerWrapper, ...>) belongs to none."""
    if _frame_kind(frame) not in ("fastokens", "hf"):
        return None
    ids = set(idents(frame[0]))
    hit = [cat for cat, keys in _STAGE_IDS if ids & keys]
    return hit[0] if len(hit) == 1 else None


def classify(frames: list[tuple[str, str]]) -> tuple[str, str | None]:
    """``(category, owning stage)`` of one stack (root..leaf). The owning
    stage is the nearest fastokens/HF stage below the leaf, for leaves that
    are generic (allocation, memcpy, hashing, regex) — where they were spent."""
    if not frames:
        return "other", None
    sym, lib = frames[-1]
    if sym.startswith("DYLD-STUB$$"):
        # a stub in the caller's binary: charge it to the function it jumps to
        sym = sym[len("DYLD-STUB$$"):]
        if _PY_API.match(sym):
            lib = "Python"
        frames = frames[:-1] + [(sym, lib)]
    if (sym == "???" or lib.startswith("<unknown")) and len(frames) > 1:
        # JIT code (PCRE2) or a stripped frame: it belongs to its caller
        return classify(frames[:-1])
    kind = _frame_kind(frames[-1])
    ids = set(idents(sym))
    names = {s for s, _ in frames}
    all_ids = set()
    for s, _ in frames:
        if s.startswith(("_R", "_ZN", "__ZN")):
            all_ids.update(idents(s))

    owner = None
    for f in reversed(frames[:-1]):
        st = _stage_of(f)
        if st:
            owner = st
            break
    if kind in ("fastokens", "hf"):
        owner = _stage_of(frames[-1]) or owner

    # blocked: GIL, thread pool, anything else
    if _BLOCKING_LEAF.search(sym) or "take_gil" in sym:
        if "take_gil" in names or any("take_gil" in s for s in names):
            return "GIL wait", owner
        if all_ids & _POOL_IDS:
            return "thread pool: blocked", owner
        return "idle / blocked (other)", owner
    # allocation (the leaf, or a generic leaf called from inside the allocator)
    in_alloc = _ALLOC.search(sym) or "libsystem_malloc" in lib or any(
        "libsystem_malloc" in l for _, l in frames[-4:-1])
    if in_alloc or (kind == "system" and any(_ALLOC.search(s) for s, _ in frames[-4:-1])):
        return "allocator (malloc/free/realloc)", owner
    if _MEMCPY.search(sym):
        return "memcpy/memmove/memset", owner
    if ids & _HASH_IDS or "siphash" in sym.lower():
        return "hashing / hash tables", owner
    if ids & _REGEX_IDS or _REGEX_C.search(sym):
        return "regex engine", owner
    if _LOCK.search(sym):
        return "locks / condvars", owner
    if _SPIN_LEAF.search(sym) or (ids & _POOL_IDS and kind != "python"):
        return "thread pool: scheduling / spin", owner
    if kind == "python":
        # Python's own code, called back from the binding (lists, Encoding objects)
        for s, l in reversed(frames[:-1]):
            k = _lib_kind(l)
            if k == "python":
                continue
            if k == "fastokens":
                if _PY_UTF8.search(sym) or any(_PY_UTF8.search(x) for x, _ in frames[-4:-1]):
                    return "binding: input str -> UTF-8", owner
                return "binding: building Python lists / objects", owner
            break
        if _PY_ALLOC.search(sym):
            return "allocator (Python obmalloc)", owner
        if _PY_FREE.search(sym) or any(_PY_FREE.search(s) for s, _ in frames[-3:-1]):
            return "Python: freeing objects (lists, Encodings)", owner
        return "Python interpreter (caller code, transformers, Jinja)", owner
    if kind == "fastokens":
        st = _stage_of(frames[-1])
        if st:
            return st, st
        if ids & {"pyo3", "_native"}:
            return "binding: Rust side (pyo3, conversions)", owner
        return ("Rust: other (fastokens)" if owner is None else owner), owner
    if kind == "hf":
        st = _stage_of(frames[-1]) or owner
        return (st or "Rust: other (HF tokenizers)"), st
    if kind == "system" and len(frames) > 1:
        # a generic libc / kernel helper (TLS lookup, clock, strlen): its caller's
        return classify(frames[:-1])
    return "other", owner


# ── summary ──────────────────────────────────────────────────────────────────

def _thread_kind(thread: str) -> str:
    t = thread.lower()
    if "main-thread" in t:
        return "main"
    if "fastokens-" in t:
        return "fastokens pool"
    if "rayon" in t:
        return "rayon pool"
    if "renderer" in t:
        return "renderer"
    return "other"


def summarize(stacks, top: int = 15) -> dict:
    """Category totals over all samples and over on-CPU samples, per thread,
    where allocation / memcpy / hashing were spent, and the hottest self
    functions."""
    total = sum(n for _, _, n in stacks)
    cats, oncpu, owner_of = Counter(), Counter(), defaultdict(Counter)
    threads = defaultdict(Counter)
    selfs = Counter()
    for thread, frames, n in stacks:
        cat, owner = classify(frames)
        cats[cat] += n
        threads[thread][cat] += n
        if cat not in BLOCKED:
            oncpu[cat] += n
            if frames:
                selfs[(short_name(frames[-1][0]), os.path.basename(frames[-1][1]))] += n
        if cat in ("allocator (malloc/free/realloc)", "memcpy/memmove/memset", "hashing / hash tables", "regex engine"):
            owner_of[cat][owner or "(outside the pipeline stages)"] += n
    busy = sum(oncpu.values())

    def share(c, d):
        return {k: {"samples": v, "share": v / d if d else None} for k, v in
                sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))}

    thr = []
    for t, c in threads.items():
        tb = sum(v for k, v in c.items() if k not in BLOCKED)
        thr.append({"thread": t, "kind": _thread_kind(t), "samples": sum(c.values()), "on_cpu": tb,
                    "top": [(k, v) for k, v in c.most_common(4)]})
    thr.sort(key=lambda r: -r["on_cpu"])
    return {
        "samples": total, "on_cpu_samples": busy,
        "categories_all": share(cats, total),
        "categories_on_cpu": share(oncpu, busy),
        "where_spent": {c: share(o, sum(o.values())) for c, o in owner_of.items()},
        "threads": thr,
        "top_self": [{"function": f, "lib": l, "samples": v, "share_on_cpu": v / busy if busy else None}
                     for (f, l), v in selfs.most_common(top)],
    }


def folded(stacks) -> str:
    """Folded stacks (``thread;frame;...;frame count``) for flame graph tools."""
    agg = Counter()
    for thread, frames, n in stacks:
        names = [thread.replace(";", ":")] + [short_name(s).replace(";", ":") for s, _ in frames]
        agg[";".join(names)] += n
    return "".join(f"{k} {v}\n" for k, v in sorted(agg.items()))


def report(summary: dict, title: str = "") -> str:
    s = summary
    lines = [f"CPU samples{(' — ' + title) if title else ''}: {s['samples']} total, {s['on_cpu_samples']} on CPU "
             f"({_pct(s['on_cpu_samples'], s['samples'])} of samples; the rest are blocked threads)", ""]
    lines.append("where the CPU goes (on-CPU samples, by the function on top of the stack):")
    for k, v in s["categories_on_cpu"].items():
        lines.append(f"  {k:58} {v['samples']:8}  {_pct1(v['share'])}")
    blocked = {k: v for k, v in s["categories_all"].items() if k in BLOCKED}
    if blocked:
        lines.append("blocked samples (share of all samples):")
        for k, v in blocked.items():
            lines.append(f"  {k:58} {v['samples']:8}  {_pct1(v['share'])}")
    for cat, owners in s["where_spent"].items():
        lines.append(f"{cat}: called from")
        for k, v in list(owners.items())[:6]:
            lines.append(f"  {k:58} {v['samples']:8}  {_pct1(v['share'])}")
    lines.append("hottest functions (self, on CPU):")
    for r in s["top_self"]:
        lines.append(f"  {_pct1(r['share_on_cpu'])}  {r['function'][:90]}  ({r['lib']})")
    lines.append("busiest threads (on-CPU samples):")
    for t in s["threads"][:8]:
        if t["on_cpu"] == 0:
            continue
        top = ", ".join(f"{k} {v}" for k, v in t["top"][:3])
        lines.append(f"  {t['thread'][:48]:48} [{t['kind']}] {t['on_cpu']:7}   {top}")
    return "\n".join(lines) + "\n"


def _pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def _pct1(x):
    return f"{100 * x:5.1f}%" if x is not None else "    -"


# ── running the sampler ──────────────────────────────────────────────────────

def sampler_available() -> str | None:
    if sys.platform == "darwin" and os.path.exists("/usr/bin/sample"):
        return "sample"
    if sys.platform.startswith("linux") and shutil.which("perf"):
        return "perf"
    return None


def sample_pid(pid: int, seconds: float, raw_path: str, interval_ms: int = 1) -> str:
    """Sample a running process; write the sampler's raw output to
    ``raw_path`` and return its text."""
    tool = sampler_available()
    if tool == "sample":
        subprocess.run(["/usr/bin/sample", str(pid), str(max(1, round(seconds))), str(interval_ms),
                        "-mayDie", "-file", raw_path], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    elif tool == "perf":  # not verified on Linux
        data = raw_path + ".data"
        subprocess.run(["perf", "record", "-F", str(max(1, 1000 // interval_ms)), "-g", "-o", data, "-p", str(pid),
                        "--", "sleep", str(seconds)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with open(raw_path, "w") as f:
            subprocess.run(["perf", "script", "-i", data], check=True, stdout=f, stderr=subprocess.DEVNULL)
    else:
        raise RuntimeError("no CPU sampler: macOS needs /usr/bin/sample; Linux needs `perf` "
                           "(e.g. linux-tools for your kernel) and permission to use it "
                           "(kernel.perf_event_paranoid <= 1 or root)")
    with open(raw_path, errors="replace") as f:
        return f.read()


def sample_command(cmd: list[str], seconds: float, raw_path: str, delay: float = 1.0, interval_ms: int = 1) -> str:
    """Start ``cmd``, wait ``delay`` seconds for it to warm up, sample it for
    ``seconds``, then stop it if it is still running."""
    p = subprocess.Popen(cmd)
    try:
        time.sleep(delay)
        if p.poll() is not None:
            raise RuntimeError(f"the command exited (code {p.returncode}) before sampling started")
        return sample_pid(p.pid, seconds, raw_path, interval_ms)
    finally:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                p.kill()


def run(out_prefix: str | None, pid: int | None = None, cmd: list[str] | None = None, from_file: str | None = None,
        seconds: float = 10.0, delay: float = 1.0, interval_ms: int = 1, title: str = "") -> dict:
    """Sample (or read an existing raw sample), classify, and write
    ``<prefix>.txt`` (report), ``.json`` (summary), ``.folded`` (folded
    stacks) and ``.raw.txt`` (the sampler's output). Returns the summary."""
    if from_file:
        with open(from_file, errors="replace") as f:
            text = f.read()
    else:
        if out_prefix:
            raw = out_prefix + ".raw.txt"
        else:
            fd, raw = tempfile.mkstemp(suffix=".sample.txt")
            os.close(fd)
        text = sample_pid(pid, seconds, raw, interval_ms) if pid else sample_command(cmd, seconds, raw, delay, interval_ms)
    stacks = parse_any(text)
    summ = summarize(stacks)
    summ["source"] = from_file or ("pid %d" % pid if pid else " ".join(cmd or []))
    rep = report(summ, title)
    if out_prefix:
        with open(out_prefix + ".txt", "w") as f:
            f.write(rep)
        with open(out_prefix + ".json", "w") as f:
            json.dump(summ, f, indent=1)
        with open(out_prefix + ".folded", "w") as f:
            f.write(folded(stacks))
    summ["report"] = rep
    return summ


# ── layers: transformers wrapper / binding / Rust / GIL / lists ─────────────

LIST_APIS = ("encoding_ids", "encoding_lists")


def layers(fn, calls: int = 1, warmup: int = 1, tokenizers=None) -> dict:
    """Run ``fn()`` ``calls`` times and split the time spent into layers:

    - ``wrapper``: everything outside the timed body of fastokens' API calls —
      the caller's own code, transformers' tokenizer classes, BatchEncoding,
      pyo3 converting the input ``str`` to UTF-8 (done before the call body
      starts; costly for text with non-ASCII characters), and freeing the
      previous call's results (its lists). ``sample()`` tells these apart;
    - ``binding_python``: inside a fastokens API call but outside Rust
      (argument conversion, building the Encoding objects);
    - ``gil_wait``: waiting to get the GIL back after Rust released it;
    - ``lists``: building Python lists from ``Encoding.ids`` and the other
      list attributes;
    - ``rust``: inside Rust, split by encode stage (``rust_stages``) and the
      rest (``rust_other``: state lock, encode_batch orchestration, ...).

    Needs the ``observe`` build. Stage times of encodes that ran on other
    threads (a multi-input ``encode_batch``) are CPU time on those threads and
    can add up to more than the caller's Rust time."""
    from . import observe as obs
    if not obs.enabled():
        raise RuntimeError("layers() needs fastokens built with the `observe` feature")
    for _ in range(warmup):
        fn()
    s0 = obs.snapshot(tokenizers)
    t0 = time.perf_counter()
    for _ in range(calls):
        fn()
    outer = (time.perf_counter() - t0) * 1e3
    an = obs.analyze(s0, obs.snapshot(tokenizers))
    api = an.get("api", {})
    tot = lambda name, key: api[name]["calls"] * api[name][key] if name in api else 0.0  # noqa: E731
    calls_api = [k for k in api if k not in LIST_APIS]
    api_ms = sum(tot(k, "mean_ms") for k in calls_api)
    rust = sum(tot(k, "rust_ms") for k in calls_api)
    gil = sum(tot(k, "gil_wait_ms") for k in calls_api)
    py = sum(tot(k, "python_ms") for k in calls_api)
    lists = sum(tot(k, "mean_ms") for k in LIST_APIS)
    stages = defaultdict(float)
    for p in an.get("paths", {}).values():
        for s, v in p["stage_ms"].items():
            stages[s] += v * p["encodes"]
        stages["other"] += p["other_ms"] * p["encodes"]
    stage_sum = sum(stages.values())
    out = {
        "calls": calls, "outer_ms": outer, "per_call_ms": outer / calls,
        "layers_ms": {"wrapper": max(0.0, outer - api_ms - lists), "binding_python": py, "gil_wait": gil,
                      "lists": lists, "rust": rust},
        "rust_stages_ms": dict(stages), "rust_other_ms": max(0.0, rust - stage_sum),
        "rust_stage_cpu_exceeds_rust_wall": stage_sum > rust * 1.05,
        "api_calls": {k: api[k]["calls"] for k in api},
        "analysis": an,
    }
    out["shares"] = {k: v / outer if outer else None for k, v in out["layers_ms"].items()}
    return out


def layers_report(r: dict, title: str = "") -> str:
    n = r["calls"]
    lines = [f"layers{(' — ' + title) if title else ''}: {n} calls, {r['per_call_ms']:.3f} ms per call", ""]
    names = {"wrapper": "outside the API calls (caller, transformers, str->UTF-8, freeing)",
             "binding_python": "fastokens binding, Python side",
             "gil_wait": "waiting for the GIL", "lists": "building Python lists (.ids, masks)", "rust": "fastokens Rust"}
    for k, v in r["layers_ms"].items():
        lines.append(f"  {names[k]:66} {v / n:9.3f} ms/call  {_pct1(r['shares'][k])}")
    lines.append("  fastokens Rust, by stage:")
    for k, v in sorted(r["rust_stages_ms"].items(), key=lambda kv: -kv[1]):
        if v > 0:
            lines.append(f"    {k:46} {v / n:9.3f} ms/call  {_pct1(v / r['outer_ms'] if r['outer_ms'] else None)}")
    lines.append(f"    {'(rest of Rust: locks, batch orchestration)':46} {r['rust_other_ms'] / n:9.3f} ms/call")
    if r["rust_stage_cpu_exceeds_rust_wall"]:
        lines.append("  note: stages ran on pool threads too; their sum is CPU time, more than the caller's Rust time")
    lines.append("  API calls: " + ", ".join(f"{k} {v}" for k, v in r["api_calls"].items()))
    return "\n".join(lines) + "\n"
