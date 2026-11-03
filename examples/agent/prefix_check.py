#!/usr/bin/env python3
"""Functional check of the prefix cache (FASTOKENS_INPUT_CACHE) on real models
with their real chat templates.

For each model, a request stream interleaves opencode agent sessions (see
`workload.py`, rendered with the model's own Jinja chat template through
`transformers.apply_chat_template`, tools included) with real multi-turn chats
(a ShareGPT sample, one request per user turn). Every request goes through
fastokens with the cache off and with several cache variants; the cached
results must equal the uncached ones (the primary check). The uncached result
is also compared with HF `tokenizers` (a secondary check: a difference there is
not caused by the cache).

Per request, beyond `encode(text)`: `add_special_tokens=True` (BOS/EOS
post-processing), `split_special_tokens=True`, `encode_ordinary`, and
periodically `encode_batch` / `encode_batch_flat`; a variant with truncation
enabled; a variant whose vocabulary is extended mid-stream with `add_tokens`;
and the patched `transformers` tokenizer (`tok(text)` and
`apply_chat_template(..., tokenize=True)`).

Usage:
  python prefix_check.py --sharegpt sharegpt400.json [--models glm-5.2,gemma-4]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import workload  # noqa: E402

# name -> (repo, renderer): "template" uses the model's chat template; a
# workload model name uses that hand-written renderer; "gemma4" is below.
MODELS = {
    "glm-5.2": ("zai-org/GLM-5.2", "template"),
    "glm-5.3": ("zai-org/GLM-5.3", "template"),
    "glm-4.7": ("zai-org/GLM-4.7", "template"),
    "qwen3-coder": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", "template"),
    "qwen3.5": ("Qwen/Qwen3.5-27B", "template"),
    "deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", "template"),
    "deepseek-v4.1-flash": ("deepseek-ai/DeepSeek-V4.1-Flash", "template"),
    "gpt-oss": ("openai/gpt-oss-120b", "template"),
    "mistral-nemo": ("mistralai/Mistral-Nemo-Instruct-2407", "template"),
    "nemotron-3-nano": ("nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16", "template"),
    "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "template"),
    "qwen2-vl": ("Qwen/Qwen2-VL-2B-Instruct", "template"),
    "minimax-m2.1": ("MiniMaxAI/MiniMax-M2.1", "minimax-m2"),
    "deepseek-v3.2": ("deepseek-ai/DeepSeek-V3.2", "deepseek-v3.1"),
    "kimi-k2.5": ("hoangquan456/Kimi-K2.5", "kimi-k2"),
    "gemma-4": ("google/gemma-4-31B-it", "gemma4"),
}

VARIANTS = [
    ("prefix", {"FASTOKENS_INPUT_CACHE": 64}),
    ("prefix-cap2", {"FASTOKENS_INPUT_CACHE": 2}),
    ("prefix+seg", {"FASTOKENS_INPUT_CACHE": 64, "FASTOKENS_SEGMENT_CACHE": 256}),
]


def hub(repo, name):
    from huggingface_hub import hf_hub_download

    try:
        return hf_hub_download(repo, name)
    except Exception:  # noqa: BLE001 - absent from the cache / repo
        return None


def chat_template(repo):
    path = hub(repo, "chat_template.jinja")
    if path:
        return open(path, encoding="utf-8").read()
    path = hub(repo, "tokenizer_config.json")
    t = json.load(open(path, encoding="utf-8")).get("chat_template") if path else None
    if isinstance(t, list):
        named = {x["name"]: x["template"] for x in t}
        t = named.get("tool_use") or named.get("default") or next(iter(named.values()))
    return t


def special_tokens(repo):
    """bos/eos/... strings from tokenizer_config.json, which templates use."""
    path = hub(repo, "tokenizer_config.json")
    cfg = json.load(open(path, encoding="utf-8")) if path else {}
    out = {}
    for k in ("bos_token", "eos_token", "unk_token", "pad_token"):
        v = cfg.get(k)
        if isinstance(v, dict):
            v = v.get("content")
        if isinstance(v, str):
            out[k] = v
    return out


def with_env(path, env):
    from fastokens._native import Tokenizer

    os.environ.update({k: str(v) for k, v in env.items()})
    try:
        return Tokenizer.from_file(path)
    finally:
        for k in env:
            os.environ.pop(k, None)


def hf_messages(msgs, n_calls=[0]):  # noqa: B006 - a counter for unique ids
    out = []
    for m in msgs:
        r = m["role"]
        if r in ("system", "user"):
            out.append({"role": r, "content": m["content"]})
        elif r == "assistant":
            a = {"role": "assistant", "content": m.get("content") or ""}
            if m.get("reasoning"):
                a["reasoning_content"] = m["reasoning"]
            if m.get("tool_calls"):
                a["tool_calls"] = [
                    {"id": f"c{abs(hash(c['id'])) % 10**8:08d}", "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in m["tool_calls"]]
            out.append(a)
        else:
            out.append({"role": "tool", "tool_call_id": f"c{abs(hash(m['tool_call_id'])) % 10**8:08d}",
                        "name": m["name"], "content": m["content"]})
    return out


def flatten_tools(msgs):
    """The same conversation with tool calls and results as plain turns, for
    templates that reject tool messages."""
    out = []
    for m in msgs:
        if m["role"] == "tool":
            m = {"role": "user", "content": f"[tool result]\n{m['content']}"}
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = "\n".join(f"call {c['function']['name']}({json.dumps(c['function']['arguments'])})"
                              for c in m["tool_calls"])
            m = {"role": "assistant", "content": (m["content"] + "\n" + calls).strip()}
        if out and out[-1]["role"] == m["role"] and m["role"] != "system":
            out[-1] = {"role": m["role"], "content": out[-1]["content"] + "\n\n" + m["content"]}
        else:
            out.append(m)
    return out


def gemma4(msgs, generation=True):
    s = "<bos>"
    for m in msgs:
        role = {"assistant": "model", "tool": "user"}.get(m["role"], m["role"])
        content = m.get("content") or ""
        for c in m.get("tool_calls") or ():
            content += f"\n<|tool_call>call:{c['name']}{json.dumps(c['arguments'])}<tool_call|>"
        if m["role"] == "tool":
            content = f"<|tool_response>{content}<tool_response|>"
        s += f"<|turn>{role}\n{content}<turn|>\n"
    return s + ("<|turn>model\n" if generation else "")


class Renderer:
    def __init__(self, name, repo, kind, tj):
        self.kind, self.name, self.stats = kind, name, {"template": 0, "template-flat": 0, "fallback": 0}
        self.tf = None
        if kind == "template":
            from transformers import PreTrainedTokenizerFast

            self.tf = PreTrainedTokenizerFast(tokenizer_file=tj, **special_tokens(repo))
            self.tf.chat_template = chat_template(repo)
            if not self.tf.chat_template:
                raise RuntimeError(f"{repo}: no chat template")
        self.tools = [{"type": "function", "function": t} for t in workload.tools()]

    def agent(self, msgs):
        if self.kind == "gemma4":
            return gemma4(msgs)
        if self.kind != "template":
            return workload.MODELS[self.kind][1](msgs)
        hf = hf_messages(msgs)
        try:
            text = self.tf.apply_chat_template(hf, tools=self.tools, tokenize=False, add_generation_prompt=True)
            self.stats["template"] += 1
            return text
        except Exception:  # noqa: BLE001 - template-specific restrictions
            pass
        try:
            text = self.tf.apply_chat_template(flatten_tools(hf), tokenize=False, add_generation_prompt=True)
            self.stats["template-flat"] += 1
            return text
        except Exception:  # noqa: BLE001
            self.stats["fallback"] += 1
            return workload.MODELS["qwen3-coder"][1](msgs)

    def chat(self, msgs):
        if self.kind == "gemma4":
            return gemma4(msgs)
        if self.kind != "template":
            return workload.MODELS[self.kind][1]([{"role": "system", "content": "You are helpful."}] + msgs)
        return self.tf.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def stream(renderer, sharegpt, n_agent, n_chat):
    """Interleaved (kind, text, messages) requests."""
    agents = [workload.session_requests(seed, compact_at=200_000, max_requests=n_agent // 3 + 1) for seed in range(3)]
    agent_reqs = []
    while len(agent_reqs) < n_agent and agents:
        for g in list(agents):
            msgs = next(g, None)
            if msgs is None:
                agents.remove(g)
            else:
                agent_reqs.append(("agent", renderer.agent(msgs), msgs))
    chat_reqs = []
    convs = [c for c in sharegpt][: max(1, n_chat // 10)]
    turn = 1
    while len(chat_reqs) < n_chat and turn < 40:
        for c in convs:
            if 2 * turn - 1 <= len(c):
                chat_reqs.append(("chat", renderer.chat(c[: 2 * turn - 1]), c[: 2 * turn - 1]))
        turn += 1
    out = []
    for i in range(max(len(agent_reqs), len(chat_reqs))):
        out += agent_reqs[i:i + 1] + chat_reqs[i:i + 1]
    return out


def check_model(name, args, sharegpt):
    import tokenizers
    from fastokens._native import Tokenizer  # noqa: F401 - fail early if missing

    repo, kind = MODELS[name]
    tj = hub(repo, "tokenizer.json")
    t0 = time.perf_counter()
    renderer = Renderer(name, repo, kind, tj)
    reqs = stream(renderer, sharegpt, args.agent, args.chat)
    ref = tokenizers.Tokenizer.from_file(tj)
    off = with_env(tj, {})
    variants = [(v, with_env(tj, env)) for v, env in VARIANTS]
    trunc_off, trunc_on = with_env(tj, {}), with_env(tj, {"FASTOKENS_INPUT_CACHE": 64})
    for t in (trunc_off, trunc_on):
        t.enable_truncation(4096, direction="left")
    add_off, add_on = with_env(tj, {}), with_env(tj, {"FASTOKENS_INPUT_CACHE": 64})

    fails, n, toks, ref_diff = [], 0, 0, 0

    def same(label, i, got, want):
        nonlocal n
        n += 1
        if list(got) != list(want):
            fails.append(f"{label} at request {i} ({reqs[i][0]}, {len(reqs[i][1])} chars)")

    for i, (k, text, _) in enumerate(reqs):
        if i == len(reqs) // 2:  # extend the vocabulary mid-stream; the cache must forget
            new = [tokenizers.AddedToken("<|probe_token|>", special=True)]
            add_off.add_tokens(new)
            add_on.add_tokens(new)
        want = off.encode(text).ids
        toks += len(want)
        for v, tok in variants:
            same(f"{v} encode", i, tok.encode(text).ids, want)
        if ref.encode(text, add_special_tokens=False).ids != list(want):
            ref_diff += 1
        if i % 3 == 0:
            w = off.encode(text, add_special_tokens=True).ids
            for v, tok in variants:
                same(f"{v} add_special_tokens", i, tok.encode(text, add_special_tokens=True).ids, w)
        if i % 5 == 0:
            w = off.encode(text, split_special_tokens=True).ids
            for v, tok in variants:
                same(f"{v} split_special_tokens", i, tok.encode(text, split_special_tokens=True).ids, w)
        if i % 10 == 0:
            w = off.encode_ordinary(text).ids
            same("prefix encode_ordinary", i, variants[0][1].encode_ordinary(text).ids, w)
        same("truncation", i, trunc_on.encode(text).ids, trunc_off.encode(text).ids)
        probe = text if i < len(reqs) // 2 else text[: len(text) // 2] + "<|probe_token|>" + text[len(text) // 2:]
        same("add_tokens", i, add_on.encode(probe).ids, add_off.encode(probe).ids)
        if i % 20 == 19:
            batch = [t for _, t, _ in reqs[i - 7: i + 1]]
            w = [e.ids for e in off.encode_batch(batch)]
            wflat = off.encode_batch_flat(batch)
            for v, tok in variants:
                same(f"{v} encode_batch", i, [e.ids for e in tok.encode_batch(batch)], w)
                same(f"{v} encode_batch_flat", i, tok.encode_batch_flat(batch), wflat)

    # The patched transformers tokenizer, cache on, against plain transformers.
    tf_n = 0
    if renderer.tf is not None:
        import fastokens
        from transformers import PreTrainedTokenizerFast

        os.environ["FASTOKENS_INPUT_CACHE"] = "64"
        fastokens.patch_transformers()
        try:
            patched = PreTrainedTokenizerFast(tokenizer_file=tj, **special_tokens(repo))
            patched.chat_template = renderer.tf.chat_template
        finally:
            fastokens.unpatch_transformers()
            os.environ.pop("FASTOKENS_INPUT_CACHE", None)
        for i, (k, text, msgs) in enumerate(reqs):
            if i % 4:
                continue
            same("transformers tok(text)", i, patched(text, add_special_tokens=False)["input_ids"],
                 renderer.tf(text, add_special_tokens=False)["input_ids"])
            tf_n += 1
            if k == "chat":
                same("transformers apply_chat_template", i,
                     patched.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True),
                     renderer.tf.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))

    sizes = [len(t) for _, t, _ in reqs]
    print(f"{name:20} {'OK  ' if not fails else 'FAIL'} {len(reqs):4} requests "
          f"(max {max(sizes) / 1e3:.0f}K chars, {toks / 1e6:.1f}M tokens), {n} comparisons, "
          f"{tf_n} via transformers, render {renderer.stats}, "
          f"uncached != HF tokenizers on {ref_diff} requests, {time.perf_counter() - t0:.0f}s", flush=True)
    for f in fails[:5]:
        print("    ", f)
    return not fails


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--sharegpt", required=True, help="JSON list of conversations ([{role, content}, ...])")
    ap.add_argument("--agent", type=int, default=150, help="agent requests per model")
    ap.add_argument("--chat", type=int, default=150, help="chat requests per model")
    args = ap.parse_args()
    sharegpt = json.load(open(args.sharegpt, encoding="utf-8"))
    ok = True
    for name in args.models.split(","):
        try:
            ok &= check_model(name, args, sharegpt)
        except Exception as e:  # noqa: BLE001 - report and go on
            print(f"{name:20} ERROR {type(e).__name__}: {e}", flush=True)
            ok = False
    print("ALL CHECKS PASSED" if ok else "CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
