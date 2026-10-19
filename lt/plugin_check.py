"""examples/vllm_observe without vLLM: register(), port selection past a busy
port, and that every tokenizer (including deep copies) is listed.
usage: FASTOKENS_SEGMENT_CACHE=16 python lt/plugin_check.py REPO"""
import copy, json, os, socket, sys, urllib.request

sys.path.insert(0, os.path.join(sys.argv[1], "examples", "vllm_observe"))
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  plugin {name}  {detail}", flush=True)


import vllm_fastokens_observe as plug  # noqa: E402

os.environ.pop("FASTOKENS_OBSERVE_PORT", None)
plug.register()
from fastokens import _compat  # noqa: E402

check("no-op without FASTOKENS_OBSERVE_PORT", not getattr(_compat._TokenizerShim, "_observe_plugin", False))
busy = socket.socket(); busy.bind(("127.0.0.1", 0)); busy.listen(1)
base = busy.getsockname()[1]
os.environ["FASTOKENS_OBSERVE_PORT"] = str(base)
plug.register(); plug.register()          # the second call must be a no-op
import fastokens  # noqa: E402

fastokens.patch_transformers()
from huggingface_hub import hf_hub_download  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(os.path.dirname(hf_hub_download("zai-org/GLM-5.2", "tokenizer.json")))
tok2 = copy.deepcopy(tok)
check("skips a busy port", plug._port is not None and plug._port != base, f"base {base} busy, got {plug._port}")
d = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{plug._port}/json").read())
check("lists the tokenizer and its deep copy", list(d["tokenizers"]) == ["tok0", "tok1"], str(list(d["tokenizers"])))
check("/json pid is this process", d["pid"] == os.getpid(), str(d["pid"]))
check("registered natives", len(plug._tokenizers) == 2, str(len(plug._tokenizers)))
tok("hello " * 50)
print(f"PLUGIN: {sum(results)} passed, {len(results) - sum(results)} failed")
