"""Tests of the CPU sampling analysis and the layer split
(`fastokens.observe.sample` / `layers`, implemented in `fastokens._profile`).

The parsing and classification run on synthetic sampler output with known
counts, in every build. One smoke test samples a real process (macOS only:
it needs /usr/bin/sample); the layer split needs the `observe` build.
"""

import base64
import json
import os
import subprocess
import sys

import pytest

import fastokens.observe as obs
from fastokens import _native, _profile
from fastokens._native import Tokenizer

ENABLED = _native.observe_enabled()

# A macOS `sample` call graph with known self counts (the count of a node
# minus its children's): a Python main thread calling fastokens, a pool
# worker, and a thread in HF `tokenizers`.
SAMPLE = """Analysis of sampling python (pid 1) every 1 millisecond
Process:         python [1]

Call graph:
    107 Thread_1   DispatchQueue_1: com.apple.main-thread  (serial)
    + 107 start  (in dyld) + 1  [0x1]
    +   107 _PyEval_EvalFrameDefault  (in Python) + 1  [0x2]
    +     67 _RNvMs_NtCs1_9fastokens9Tokenizer6encode  (in _native.abi3.so) + 1  [0x3]
    +     ! 30 _RNvMs_NtNtCs1_9fastokens6models3bpeNtB2_3Bpe5merge  (in _native.abi3.so) + 1  [0x4]
    +     ! : 10 _xzm_xzone_malloc_tiny  (in libsystem_malloc.dylib) + 1  [0x5]
    +     ! : | 4 mach_absolute_time  (in libsystem_kernel.dylib) + 1  [0x6]
    +     ! : 6 _platform_memmove  (in libsystem_platform.dylib) + 1  [0x7]
    +     ! 15 _RNvXs_NtNtCs1_9fastokens14pre_tokenizers5splitNtB4_5Split12pre_tokenize  (in _native.abi3.so) + 1  [0x8]
    +     ! : 8 pcre2_jit_match_8  (in _native.abi3.so) + 1  [0x9]
    +     ! :   5 ???  (in <unknown binary>)  [0xa]
    +     ! 5 PyList_New  (in Python) + 1  [0xb]
    +     ! 3 DYLD-STUB$$Py_IncRef  (in _native.abi3.so) + 1  [0x17]
    +     ! 4 unicode_encode_utf8  (in Python) + 1  [0x18]
    +     !   4 ucs4lib_utf8_encoder  (in Python) + 1  [0x19]
    +     ! 5 take_gil  (in Python) + 1  [0xc]
    +     !   5 _pthread_cond_wait  (in libsystem_pthread.dylib) + 1  [0xd]
    +     !     5 __psynch_cvwait  (in libsystem_kernel.dylib) + 1  [0xe]
    +     20 list_dealloc  (in Python) + 1  [0xf]
    +     5 _PyObject_Malloc  (in Python) + 1  [0x10]
    50 Thread_2: fastokens-0
    + 50 thread_start  (in libsystem_pthread.dylib) + 1  [0x11]
    +   50 _RNvMs_NtCs1_10rayon_core8registryNtB2_12WorkerThread15wait_until_cold  (in _native.abi3.so) + 1  [0x12]
    +     40 __psynch_cvwait  (in libsystem_kernel.dylib) + 1  [0x13]
    +     5 swtch_pri  (in libsystem_kernel.dylib) + 1  [0x14]
    20 Thread_3
    + 20 _RNvMs3_NtNtNtCs2_10tokenizers6models3bpe5modelNtB5_3BPE10merge_word  (in tokenizers.abi3.so) + 1  [0x15]
    +   6 _RINvYNtNtNtCs3_3std4hash6random11RandomStateNtNtCs4_4core4hash11BuildHasher8hash_oneRjECs2_10tokenizers  (in tokenizers.abi3.so) + 1  [0x16]

Total number in stack (recursive counted multiple, when >=5):

Sort by top of stack, same collapsed (when >= 5):
        __psynch_cvwait  (in libsystem_kernel.dylib)        45

Binary Images:
"""

EXPECTED_ON_CPU = {
    "Rust: other (fastokens)": 5,
    "stage: BPE": 14 + 14,
    "allocator (malloc/free/realloc)": 10,
    "memcpy/memmove/memset": 6,
    "stage: pre-tokenize": 7,
    "regex engine": 8,
    "binding: building Python lists / objects": 5 + 3,
    "binding: input str -> UTF-8": 4,
    "Python: freeing objects (lists, Encodings)": 20,
    "allocator (Python obmalloc)": 5,
    "Python interpreter (caller code, transformers, Jinja)": 15,
    "thread pool: scheduling / spin": 10,
    "hashing / hash tables": 6,
}
EXPECTED_BLOCKED = {"GIL wait": 5, "thread pool: blocked": 40}

PERF = """python 100/101 1.000: 1 cpu-clock:
\t    1000 _platform_memmove+0x10 (/usr/lib/libc.so.6)
\t    2000 _ZN9fastokens6models3bpe5merge17h0123456789abcdefE+0x20 (/x/_native.abi3.so)
\t    3000 _PyEval_EvalFrameDefault+0x1 (/usr/lib/libpython3.12.so.1.0)

python 100/102 1.001: 1 cpu-clock:
\t    2000 _ZN9fastokens6models3bpe5merge17h0123456789abcdefE+0x20 (/x/_native.abi3.so)
\t    3000 _PyEval_EvalFrameDefault+0x1 (/usr/lib/libpython3.12.so.1.0)

"""


def test_idents_reads_v0_and_legacy_rust_symbols():
    assert _profile.idents("_RNvMs3_NtNtNtCs2_10tokenizers6models3bpe5modelNtB5_3BPE10merge_word") == [
        "tokenizers", "models", "bpe", "model", "BPE", "merge_word"]
    assert _profile.idents("_RNvMs_NtCs1_9fastokens9Tokenizer6encode") == ["fastokens", "Tokenizer", "encode"]
    assert _profile.idents("_ZN9fastokens6models3bpe5merge17h0123456789abcdefE") == [
        "fastokens", "models", "bpe", "merge"]
    assert _profile.idents("_platform_memmove") == ["_platform_memmove"]
    assert _profile.short_name("_RNvMs_NtCs1_9fastokens9Tokenizer6encode") == "fastokens::Tokenizer::encode"


def test_parse_macos_sample_gives_self_counts_per_thread():
    stacks = _profile.parse_macos_sample(SAMPLE)
    assert sum(n for _, _, n in stacks) == 177
    by_thread = {}
    for t, _, n in stacks:
        by_thread[t] = by_thread.get(t, 0) + n
    assert by_thread == {"Thread_1 DispatchQueue_1: com.apple.main-thread  (serial)": 107,
                         "Thread_2: fastokens-0": 50, "Thread_3": 20}
    # the BPE frame keeps 30 - 10 - 6 = 14 samples of its own
    bpe = [n for _, fr, n in stacks if fr[-1][0].endswith("3Bpe5merge")]
    assert bpe == [14]
    leaf = [fr for _, fr, _ in stacks if fr[-1][0] == "mach_absolute_time"][0]
    assert [f[0] for f in leaf[:3]] == ["start", "_PyEval_EvalFrameDefault", "_RNvMs_NtCs1_9fastokens9Tokenizer6encode"]


def test_classification_categories_and_where_generic_work_was_spent():
    s = _profile.summarize(_profile.parse_macos_sample(SAMPLE))
    assert s["samples"] == 177
    assert s["on_cpu_samples"] == 132
    got = {k: v["samples"] for k, v in s["categories_on_cpu"].items()}
    assert got == EXPECTED_ON_CPU
    blocked = {k: v["samples"] for k, v in s["categories_all"].items() if k in _profile.BLOCKED}
    assert blocked == EXPECTED_BLOCKED
    assert abs(sum(v["share"] for v in s["categories_on_cpu"].values()) - 1) < 1e-9
    # mach_absolute_time inside the allocator is allocation; both are spent in BPE
    assert s["where_spent"]["allocator (malloc/free/realloc)"]["stage: BPE"]["samples"] == 10
    assert s["where_spent"]["memcpy/memmove/memset"]["stage: BPE"]["samples"] == 6
    # PCRE2 JIT code (an unknown frame) belongs to its caller, inside pre-tokenization
    assert s["where_spent"]["regex engine"]["stage: pre-tokenize"]["samples"] == 8
    # HF tokenizers' hashing, inside its BPE
    assert s["where_spent"]["hashing / hash tables"]["stage: BPE"]["samples"] == 6
    kinds = {t["thread"]: t["kind"] for t in s["threads"]}
    assert kinds["Thread_2: fastokens-0"] == "fastokens pool"
    assert s["threads"][0]["thread"].startswith("Thread_1")
    assert _profile.report(s)  # renders


def test_a_generic_over_every_stage_belongs_to_none():
    frame = ("_RNvMsi_NtCs5_10tokenizers9tokenizerINtB5_13TokenizerImplNtNtB7_6models12ModelWrapper"
             "NtNtB7_11normalizers17NormalizerWrapperNtNtB7_14pre_tokenizers19PreTokenizerWrapperE6encode",
             "tokenizers.abi3.so")
    assert _profile.classify([frame]) == ("Rust: other (HF tokenizers)", None)


def test_parse_perf_script():
    stacks = _profile.parse_perf_script(PERF)
    assert [(t, len(fr), n) for t, fr, n in stacks] == [("python/101", 3, 1), ("python/102", 2, 1)]
    s = _profile.summarize(stacks)
    assert {k: v["samples"] for k, v in s["categories_on_cpu"].items()} == {
        "memcpy/memmove/memset": 1, "stage: BPE": 1}
    assert s["where_spent"]["memcpy/memmove/memset"]["stage: BPE"]["samples"] == 1


def test_folded_stacks_keep_every_sample():
    stacks = _profile.parse_macos_sample(SAMPLE)
    lines = _profile.folded(stacks).splitlines()
    assert sum(int(line.rsplit(" ", 1)[1]) for line in lines) == 177
    assert any(line.startswith("Thread_2: fastokens-0;thread_start;rayon_core::registry::WorkerThread::wait_until_cold")
               for line in lines)


def test_sample_command_reads_an_existing_report(tmp_path):
    raw = tmp_path / "raw.txt"
    raw.write_text(SAMPLE)
    out = tmp_path / "prof"
    r = subprocess.run([sys.executable, "-m", "fastokens.observe", "sample", "--from", str(raw), "--out", str(out)],
                       capture_output=True, text=True, check=True)
    assert "where the CPU goes" in r.stdout and "stage: BPE" in r.stdout
    summ = json.loads((tmp_path / "prof.json").read_text())
    assert summ["on_cpu_samples"] == 132
    assert (tmp_path / "prof.folded").read_text().count("\n") > 5
    assert "allocator" in (tmp_path / "prof.txt").read_text()
    r = subprocess.run([sys.executable, "-m", "fastokens.observe", "sample"], capture_output=True, text=True)
    assert r.returncode != 0 and "needs --pid" in r.stderr


def _scanner_model(tmp_path):
    ranks = [bytes([i]) for i in range(256)] + [b"th", b"the", b" the", b"in", b"ing", b" th", b"he", b"ll", b"hell", b"hello"]
    path = tmp_path / "scan.model"
    path.write_text("".join(f"{base64.b64encode(t).decode()} {i}\n" for i, t in enumerate(ranks)))
    return path


@pytest.mark.skipif(_profile.sampler_available() != "sample", reason="needs macOS /usr/bin/sample")
def test_sampling_a_running_encode_loop(tmp_path):
    model = _scanner_model(tmp_path)
    code = (
        "import fastokens._native as n\n"
        f"t = n.Tokenizer.from_tiktoken({str(model)!r}, encoding='o200k_base', special_tokens={{'<|im_start|>': 400}})\n"
        "text = ('hello the thing ' * 4000 + '<|im_start|>') * 8\n"
        "while True:\n"
        "    t.encode(text).ids\n"
    )
    out = tmp_path / "live"
    r = subprocess.run([sys.executable, "-m", "fastokens.observe", "sample", "--seconds", "1", "--delay", "0.5",
                        "--out", str(out), "--", sys.executable, "-c", code],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    summ = json.loads((tmp_path / "live.json").read_text())
    assert summ["on_cpu_samples"] > 50
    cats = summ["categories_on_cpu"]
    fastokens_work = sum(v["samples"] for k, v in cats.items()
                         if k.startswith("stage:") or k in ("Rust: other (fastokens)", "regex engine",
                                                            "memcpy/memmove/memset", "allocator (malloc/free/realloc)",
                                                            "binding: building Python lists / objects",
                                                            "binding: input str -> UTF-8"))
    assert fastokens_work > 0
    assert (tmp_path / "live.folded").exists() and (tmp_path / "live.raw.txt").exists()


def _tiny_json():
    return {"added_tokens": [{"id": 5, "content": "<end>", "special": True, "single_word": False,
                              "lstrip": False, "rstrip": False, "normalized": False}],
            "model": {"type": "BPE", "vocab": {"<": 0, "e": 1, "n": 2, "d": 3, ">": 4}, "merges": []}}


@pytest.mark.skipif(ENABLED, reason="checks the message of a build without observe")
def test_layers_needs_the_observe_build():
    tok = Tokenizer.from_json_str(json.dumps(_tiny_json()))
    with pytest.raises(RuntimeError, match="observe"):
        obs.layers(lambda: tok.encode("end").ids)


@pytest.mark.skipif(not ENABLED, reason="build without the observe feature")
def test_layers_adds_up_to_the_outer_time():
    tok = Tokenizer.from_json_str(json.dumps(_tiny_json()))
    text = "end<end>dne" * 300
    r = obs.layers(lambda: tok.encode(text).ids, calls=50)
    assert r["calls"] == 50 and r["api_calls"]["encode"] == 50 and r["api_calls"]["encoding_ids"] == 50
    lay = r["layers_ms"]
    assert set(lay) == {"wrapper", "binding_python", "gil_wait", "lists", "rust"}
    assert abs(sum(lay.values()) - r["outer_ms"]) < 1e-6 * max(1, r["outer_ms"]) or lay["wrapper"] == 0
    assert lay["rust"] > 0 and lay["lists"] > 0
    assert r["rust_stages_ms"]["model"] > 0 and r["rust_stages_ms"]["added_tokens"] > 0
    assert "fastokens Rust, by stage" in obs.layers_report(r)


@pytest.mark.skipif(not ENABLED, reason="build without the observe feature")
def test_layers_command(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps(_tiny_json()))
    (tmp_path / "inputs.json").write_text(json.dumps([["end<end>", "den" * 50], "nde"]))
    r = subprocess.run([sys.executable, "-m", "fastokens.observe", "layers", "--tokenizer", str(tmp_path / "tokenizer.json"),
                        "--inputs", str(tmp_path / "inputs.json"), "--calls", "30"],
                       capture_output=True, text=True, check=True, env={**os.environ})
    assert "30 calls" in r.stdout and "building Python lists" in r.stdout
    r = subprocess.run([sys.executable, "-m", "fastokens.observe", "layers", "--tokenizer", str(tmp_path / "tokenizer.json"),
                        "--inputs", str(tmp_path / "inputs.json"), "--json"],
                       capture_output=True, text=True, check=True)
    d = json.loads(r.stdout)
    assert d["calls"] == 3 and d["api_calls"]["encode"] == 3
