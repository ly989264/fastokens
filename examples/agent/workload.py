"""Deterministic opencode-style agent workload for the correctness and
performance harness (`check.py`, `bench.py`).

What an opencode session sends to the model server
---------------------------------------------------
The chat API is stateless, so every agent step re-sends the *whole* history:

    request 1:  [system][tools][user]
    request 2:  [system][tools][user][assistant: tool call][tool result]
    request 3:  ... previous request ... [assistant: tool call][tool result]

The server renders that history with the model's chat template (which wraps
every message in special tokens such as `<|im_start|>`) and tokenizes the full
string. This module reproduces that stream:

* a fixed system prompt (~20 KB, incl. an AGENTS.md) and 14 tool schemas
  (~25 KB), identical across sessions, like one opencode deployment;
* a loop of reasoning text, tool calls (read / grep / glob / bash / edit /
  write / todowrite / webfetch, sometimes several in parallel) and tool
  results in opencode's own formats (`00001| ` numbered file reads, grep
  listings, cargo/pytest logs, diffs), with English and Chinese user turns;
* chat-template behaviour that rewrites history: most templates drop the
  reasoning of turns before the latest user message;
* opencode's auto-compaction: once the history is large it is replaced by a
  summary and the session continues.

File contents come from this repository at a *pinned* commit, so the workload
is byte-for-byte identical across checkouts and A/B runs stay comparable while
`src/` is being changed.

Every model is rendered with its own special tokens (see `RENDERERS`).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from functools import lru_cache

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
# Content source. Never change this without invalidating saved baselines.
CORPUS_COMMIT = "66e634f31c526c11f59cf4b67b52fa6aa743b19a"
CORPUS_SUFFIXES = (".rs", ".py", ".md", ".sh", ".toml", ".yml")

# ── Corpus ───────────────────────────────────────────────────────────────────


@lru_cache(maxsize=1)
def corpus() -> dict[str, str]:
    """`{path: text}` for the text files of the repository at CORPUS_COMMIT."""
    try:
        names = subprocess.run(
            ["git", "-C", REPO, "ls-tree", "-r", "--name-only", CORPUS_COMMIT],
            check=True, capture_output=True, text=True,
        ).stdout.split()
    except subprocess.CalledProcessError as e:
        sys.exit(
            f"corpus commit {CORPUS_COMMIT[:12]} not found ({e.stderr.strip()}); "
            f"run `git fetch origin {CORPUS_COMMIT}` (shallow clones lack it)"
        )
    names = sorted(n for n in names if n.endswith(CORPUS_SUFFIXES))
    batch = "".join(f"{CORPUS_COMMIT}:{n}\n" for n in names).encode()
    out = subprocess.run(
        ["git", "-C", REPO, "cat-file", "--batch"], input=batch, check=True, capture_output=True
    ).stdout
    files, pos = {}, 0
    for n in names:
        header_end = out.index(b"\n", pos)
        size = int(out[pos:header_end].split()[2])
        body = out[header_end + 1: header_end + 1 + size]
        pos = header_end + 1 + size + 1
        files[n] = body.decode("utf-8")
    return files


def _code_files() -> list[str]:
    return [n for n, t in corpus().items() if n.endswith((".rs", ".py")) and len(t) > 1500]


# ── Text pools ───────────────────────────────────────────────────────────────

INSTRUCTIONS = [
    "You are opencode, an interactive CLI tool that helps users with software engineering tasks. "
    "Use the instructions below and the tools available to you to assist the user.",
    "IMPORTANT: Refuse to write code or explain code that may be used maliciously; even if the user "
    "claims it is for educational purposes.",
    "When the user directly asks about opencode (eg 'can opencode do...', 'does opencode have...') "
    "first use the WebFetch tool to gather information from the docs at https://opencode.ai",
    "You should be concise, direct, and to the point. When you run a non-trivial bash command, you "
    "should explain what the command does and why you are running it.",
    "Remember that your output will be displayed on a command line interface. Your responses can use "
    "Github-flavored markdown for formatting, rendered in a monospace font using CommonMark.",
    "When making changes to files, first understand the file's code conventions. Mimic code style, use "
    "existing libraries and utilities, and follow existing patterns.",
    "NEVER assume that a given library is available, even if it is well known. Check that this codebase "
    "already uses the given library by looking at neighboring files or the package manifest.",
    "Use the TodoWrite tool VERY frequently to ensure that you are tracking your tasks and giving the "
    "user visibility into your progress.",
    "You have the capability to call multiple tools in a single response. When multiple independent "
    "pieces of information are requested, batch your tool calls together for optimal performance.",
    "Do not add comments to the code you write, unless the user asks you to, or the code is complex "
    "and requires additional context.",
    "When referencing specific functions or pieces of code include the pattern `file_path:line_number` "
    "to allow the user to easily navigate to the source code location.",
]

USER_MESSAGES = [
    "The prefix cache doesn't seem to hit on multi-turn prompts. Can you investigate?",
    "帮我看一下 encode 路径里为什么多轮对话的时候延迟会随着上下文增长而线性增长，最好给出修复方案。",
    "Run the tests again and fix any failures.",
    "请把这个函数重构一下，保持行为不变，并补充单元测试。",
    "Looks good. Now update the README to mention the new option.",
    "Why is `tokenize_batched` slower than `tokenize` on long inputs? Profile it and explain.",
    "把 PCRE2 的并行切块逻辑解释给我听，特别是跨块边界的匹配是怎么修复的。",
    "Add a regression test for the chunk-boundary bug we just fixed, then commit.",
]

REASONING = [
    "Let me look at how this is implemented before changing anything.",
    "The failing assertion compares token ids across a chunk boundary, so the bug is likely in the merge step.",
    "我先读一下相关代码，确认前缀缓存的命中逻辑。",
    "Now I understand the structure; next I'll grep for callers of this function.",
    "The user wants behaviour preserved, so I should run the existing tests first to get a baseline.",
    "这里的锁在整个 LCP 扫描期间都持有，并发场景下可能成为瓶颈。",
    "I'll check the test results and then decide whether the fix is complete.",
    "There are two call sites; both pass a borrowed buffer, so the lifetime change is safe.",
]

FINAL_ANSWERS = [
    "I've made the change and all tests pass. Summary:\n\n- Fixed the boundary check in `merge_chunk_matches`\n"
    "- Added a regression test covering matches that span three chunks\n\nLet me know if you want me to open a PR.",
    "完成了。主要改动：\n\n1. 修复了缓存命中后不回写的问题\n2. 增加了多轮对话的测试用例\n\n所有测试都已通过。",
    "The slowdown comes from re-tokenizing the full history on every request. See `src/lib.rs:1172` for "
    "the fast-path condition that multi-segment inputs never satisfy.",
]

TOOL_NAMES = ["bash", "edit", "webfetch", "glob", "grep", "list", "patch", "read",
              "write", "todowrite", "todoread", "task", "multiedit", "invalid"]


# ── Shared prefix: system prompt + tool schemas (identical for all sessions) ─


@lru_cache(maxsize=1)
def system_prompt() -> str:
    rng = random.Random(0)
    body = "\n".join(rng.choice(INSTRUCTIONS) for _ in range(40))
    env = ("<env>\n  Working directory: /home/dev/fastokens\n  Is directory a git repo: yes\n"
           "  Platform: linux\n  Today's date: Thu Sep 24 2026\n</env>\n<project>\n"
           + "\n".join(sorted(corpus())[:60]) + "\n</project>")
    agents_md = corpus()["README.md"]
    return f"{body}\n\n{env}\n\n# AGENTS.md\n\n{agents_md}"


@lru_cache(maxsize=1)
def tools() -> list[dict]:
    rng = random.Random(1)
    out = []
    for name in TOOL_NAMES:
        desc = f"Executes the {name} operation.\n\nUsage notes:\n" + "\n".join(
            f"  - {rng.choice(INSTRUCTIONS)}" for _ in range(rng.randint(3, 14)))
        props = {}
        for key in ["filePath", "pattern", "content", "timeout", "description", "limit"][: rng.randint(2, 6)]:
            props[key] = {"type": rng.choice(["string", "number", "boolean"]),
                          "description": rng.choice(INSTRUCTIONS)[: rng.randint(40, 200)]}
        params = {"type": "object", "properties": props, "required": [next(iter(props))],
                  "additionalProperties": False, "$schema": "http://json-schema.org/draft-07/schema#"}
        out.append({"name": name, "description": desc, "parameters": params})
    return out


# ── Tool results in opencode's formats ───────────────────────────────────────


def _read_result(rng, path):
    lines = corpus()[path].split("\n")
    offset = rng.randint(0, max(0, len(lines) - 200))
    chunk = lines[offset: offset + rng.choice([200, 400, 800, 2000])]
    body = "\n".join(f"{offset + i + 1:05d}| {line[:2000]}" for i, line in enumerate(chunk))
    end = offset + len(chunk)
    tail = (f"\n\n(File has more lines. Use 'offset' parameter to read beyond line {end})"
            if end < len(lines) else f"\n\n(End of file - total {len(lines)} lines)")
    return f"<file>\n{body}{tail}\n</file>"


def _grep_result(rng, files):
    out = []
    for f in rng.sample(files, min(8, len(files))):
        hits = [(i, l) for i, l in enumerate(corpus()[f].split("\n")) if "fn " in l or "def " in l]
        hits = hits[: rng.randint(3, 25)]
        out.append(f"/home/dev/fastokens/{f}:")
        out += [f"  Line {i + 1}: {l.strip()[:200]}" for i, l in hits]
    return f"Found {len(out)} matches\n" + "\n".join(out)


def _glob_result(rng):
    names = sorted(corpus())
    return "\n".join(f"/home/dev/fastokens/{n}" for n in rng.sample(names, min(len(names), rng.randint(5, 40))))


def _bash_result(rng, files):
    kind = rng.choice(["cargo", "pytest", "diff", "status"])
    if kind == "cargo":
        n = rng.randint(20, 300)
        out = ["   Compiling fastokens v0.3.2 (/home/dev/fastokens)",
               "    Finished `test` profile [unoptimized + debuginfo] target(s) in 14.21s",
               "     Running unittests src/lib.rs (target/debug/deps/fastokens-3f2a9c1d0e8b7a64)",
               "", f"running {n} tests"]
        for i in range(n):
            mod = rng.choice(["pre_tokenizers::split::tests", "models::bpe::tests", "local_tests", "decoders::tests"])
            out.append(f"test {mod}::case_{i:04d}_{rng.choice(['roundtrip', 'boundary', 'unicode', 'cache'])} ... ok")
        out.append(f"\ntest result: ok. {n} passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 3.07s")
        return "\n".join(out)
    if kind == "pytest":
        n = rng.randint(10, 80)
        dots = "".join(rng.choice("....F.") for _ in range(n))
        return (f"============================= test session starts ==============================\n"
                f"platform linux -- Python 3.12.3, pytest-8.3.2\ncollected {n} items\n\n"
                f"python/tests/test_encode_segments.py {dots}\n\n"
                f"=========================== short test summary info ============================\n"
                f"FAILED python/tests/test_encode_segments.py::test_trust_boundary - AssertionError\n"
                f"========================= 1 failed, {n - 1} passed in 4.12s =========================")
    if kind == "diff":
        f = rng.choice(files)
        lines = corpus()[f].split("\n")
        a = rng.randint(0, max(0, len(lines) - 40))
        hunk = []
        for line in lines[a: a + 30]:
            r = rng.random()
            hunk.append(("-" if r < 0.1 else "+" if r < 0.2 else " ") + line)
        return (f"diff --git a/{f} b/{f}\nindex 3f2a9c1..0e8b7a6 100644\n--- a/{f}\n+++ b/{f}\n"
                f"@@ -{a + 1},30 +{a + 1},30 @@\n" + "\n".join(hunk))
    return ("On branch perf/segment-cache\nChanges not staged for commit:\n"
            + "\n".join(f"\tmodified:   {f}" for f in rng.sample(files, 3))
            + "\n\nno changes added to commit (use \"git add\" and/or \"git commit -a\")")


def _tool_call(rng, files):
    """One (name, arguments, result) triple."""
    kind = rng.choices(["read", "grep", "glob", "bash", "edit", "write", "todowrite", "webfetch"],
                       [30, 12, 4, 14, 10, 3, 4, 2])[0]
    path = rng.choice(files)
    fp = f"/home/dev/fastokens/{path}"
    if kind == "read":
        return "read", {"filePath": fp}, _read_result(rng, path)
    if kind == "grep":
        return "grep", {"pattern": r"fn (encode|decode)_\w+", "path": "/home/dev/fastokens/src",
                        "include": "*.rs"}, _grep_result(rng, files)
    if kind == "glob":
        return "glob", {"pattern": "**/*.rs"}, _glob_result(rng)
    if kind == "bash":
        cmd = rng.choice(["cargo test --release 2>&1 | tail -n 400", "pytest -q python/tests",
                          "git diff", "git status"])
        return "bash", {"command": cmd, "description": "Run the command"}, _bash_result(rng, files)
    text = corpus()[path]
    if kind == "edit":
        a = rng.randint(0, max(0, len(text) - 1500))
        old = text[a: a + rng.randint(200, 1200)]
        return ("edit", {"filePath": fp, "oldString": old, "newString": old.replace("self", "this")},
                "Edit applied successfully.\n\n<file_diagnostics>\nNo errors found.\n</file_diagnostics>")
    if kind == "write":
        a = rng.randint(0, max(0, len(text) - 6000))
        return "write", {"filePath": fp + ".new", "content": text[a: a + 6000]}, "File written successfully."
    if kind == "todowrite":
        todos = [{"id": str(i), "content": rng.choice(REASONING), "status": rng.choice(["pending", "in_progress", "completed"]),
                  "priority": rng.choice(["high", "medium", "low"])} for i in range(rng.randint(2, 8))]
        return "todowrite", {"todos": todos}, json.dumps(todos, indent=2, ensure_ascii=False)
    return ("webfetch", {"url": "https://opencode.ai/docs/config", "format": "markdown"},
            corpus()["README.md"][: rng.randint(2000, 7000)])


# ── Sessions ─────────────────────────────────────────────────────────────────


def _content_len(m) -> int:
    n = len(m.get("content") or "") + len(m.get("reasoning") or "")
    for c in m.get("tool_calls") or ():
        n += len(json.dumps(c["arguments"]))
    return n


def session_requests(seed: int, max_requests: int = 10_000, compact_at: int = 700_000,
                     stop_after_compactions: int = 1):
    """Yield the message list of every request an opencode session sends.

    A request is sent after each user message and after each batch of tool
    results. When the history's content exceeds `compact_at` characters it is
    replaced by a summary (opencode's auto-compaction); the session ends after
    `stop_after_compactions` compactions plus a short continuation, or after
    `max_requests` requests.
    """
    rng = random.Random(seed)
    files = _code_files()
    history = [{"role": "system", "content": system_prompt()},
               {"role": "user", "content": rng.choice(USER_MESSAGES)}]
    size = sum(map(_content_len, history))
    sent = compactions = 0
    after_compaction = 0
    call_id = 0
    while sent < max_requests:
        yield list(history)
        sent += 1
        if compactions >= stop_after_compactions:
            after_compaction += 1
            if after_compaction > 25:
                return
        if rng.random() < 0.08:  # the assistant finishes; the user follows up
            msgs = [{"role": "assistant", "content": rng.choice(FINAL_ANSWERS),
                     "reasoning": rng.choice(REASONING)},
                    {"role": "user", "content": rng.choice(USER_MESSAGES)}]
        else:
            calls, results = [], []
            for _ in range(1 if rng.random() < 0.8 else rng.randint(2, 3)):
                name, args, result = _tool_call(rng, files)
                calls.append({"id": f"call_{call_id}", "name": name, "arguments": args})
                results.append({"role": "tool", "tool_call_id": f"call_{call_id}", "name": name, "content": result})
                call_id += 1
            text = rng.choice(REASONING) if rng.random() < 0.3 else ""
            reasoning = " ".join(rng.choice(REASONING) for _ in range(rng.randint(1, 5)))
            msgs = [{"role": "assistant", "content": text, "reasoning": reasoning, "tool_calls": calls}] + results
        history += msgs
        size += sum(map(_content_len, msgs))
        if size > compact_at:
            compactions += 1
            summary = ("Summary of the conversation so far (auto-compacted):\n\n"
                       + "\n".join(f"- {rng.choice(REASONING)}" for _ in range(60))
                       + "\n\nFiles touched:\n" + "\n".join(f"- {f}" for f in rng.sample(files, 12)))
            history = [history[0], {"role": "user", "content": summary},
                       {"role": "user", "content": rng.choice(USER_MESSAGES)}]
            size = sum(map(_content_len, history))


# ── Chat-template renderers ──────────────────────────────────────────────────
#
# These follow the structure of each model's published chat template closely
# enough that special tokens sit where the real template puts them. The rule
# most templates share: reasoning is kept only for assistant turns after the
# last user message (earlier reasoning is dropped, rewriting history).


def _last_user(msgs) -> int:
    return max(i for i, m in enumerate(msgs) if m["role"] == "user")


def _keep_reasoning(msgs, i, policy) -> bool:
    return policy == "all" or (policy == "current" and i > _last_user(msgs))


def _qwen3_coder(msgs, keep="current"):
    tool_xml = "\n".join(
        f"<function>\n<name>{t['name']}</name>\n<description>{t['description']}</description>\n<parameters>\n"
        + "".join(f"<parameter>\n<name>{k}</name>\n<type>{v['type']}</type>\n<description>{v['description']}</description>\n</parameter>\n"
                  for k, v in t["parameters"]["properties"].items())
        + "</parameters>\n</function>" for t in tools())
    s = (f"<|im_start|>system\n{msgs[0]['content']}\n\n# Tools\n\nYou have access to the following functions:\n\n"
         f"<tools>\n{tool_xml}\n</tools>\n\nIf you choose to call a function ONLY reply in the following format "
         f"with NO suffix:\n\n<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\n"
         f"value_1\n</parameter>\n</function>\n</tool_call><|im_end|>\n")
    prev = None
    for i, m in enumerate(msgs[1:], 1):
        r = m["role"]
        if r == "user":
            s += f"<|im_start|>user\n{m['content']}<|im_end|>\n"
        elif r == "assistant":
            s += f"<|im_start|>assistant\n{m['content']}"
            for c in m.get("tool_calls") or ():
                params = "".join(f"<parameter={k}>\n{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}\n</parameter>\n"
                                 for k, v in c["arguments"].items())
                s += f"\n<tool_call>\n<function={c['name']}>\n{params}</function>\n</tool_call>"
            s += "<|im_end|>\n"
        else:
            if prev != "tool":
                s += "<|im_start|>user"
            s += f"\n<tool_response>\n{m['content']}\n</tool_response>"
            if i == len(msgs) - 1 or msgs[i + 1]["role"] != "tool":
                s += "<|im_end|>\n"
        prev = r
    return s + "<|im_start|>assistant\n"


def _glm(msgs, keep="current"):
    tj = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools())
    s = (f"[gMASK]<sop><|system|>\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
         f"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tj}\n</tools>\n\n"
         f"{msgs[0]['content']}")
    prev = None
    for i, m in enumerate(msgs[1:], 1):
        r = m["role"]
        if r == "user":
            s += f"<|user|>\n{m['content']}"
        elif r == "assistant":
            think = m.get("reasoning", "") if _keep_reasoning(msgs, i, keep) else ""
            s += f"<|assistant|>\n<think>{think}</think>\n{m['content']}"
            for c in m.get("tool_calls") or ():
                args = "".join(f"<arg_key>{k}</arg_key>\n<arg_value>{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</arg_value>\n"
                               for k, v in c["arguments"].items())
                s += f"\n<tool_call>{c['name']}\n{args}</tool_call>"
        else:
            if prev != "tool":
                s += "<|observation|>"
            s += f"\n<tool_response>\n{m['content']}\n</tool_response>"
        prev = r
    return s + "<|assistant|>\n"


def _glm5(msgs, keep="current"):
    """GLM-5.x (checked against GLM-5.2's chat_template.jinja): no newline after
    the role tokens, a reasoning-effort system line, `<think></think>` on every
    assistant turn, and tool results grouped after one `<|observation|>`."""
    tj = "\n".join(json.dumps(t, ensure_ascii=False) for t in tools())
    s = ("[gMASK]<sop><|system|>Reasoning Effort: Max<|system|>\n# Tools\n\n"
         "You may call one or more functions to assist with the user query.\n\n"
         f"You are provided with function signatures within <tools></tools> XML tags:\n<tools>\n{tj}\n</tools>\n\n"
         "For each function call, output the function name and arguments within the following XML format:\n"
         "<tool_call>{function-name}<arg_key>{arg-key-1}</arg_key><arg_value>{arg-value-1}</arg_value>"
         "<arg_key>{arg-key-2}</arg_key><arg_value>{arg-value-2}</arg_value>...</tool_call>"
         f"<|system|>{msgs[0]['content']}")
    prev = None
    for i, m in enumerate(msgs[1:], 1):
        r = m["role"]
        if r == "user":
            s += f"<|user|>{m['content']}"
        elif r == "assistant":
            think = m.get("reasoning", "") if _keep_reasoning(msgs, i, keep) else ""
            s += f"<|assistant|><think>{think}</think>{m['content'].strip()}"
            for c in m.get("tool_calls") or ():
                args = "".join(f"<arg_key>{k}</arg_key><arg_value>{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</arg_value>"
                               for k, v in c["arguments"].items())
                s += f"\n<tool_call>{c['name']}{args}</tool_call>"
        else:
            if prev != "tool":
                s += "<|observation|>"
            s += f"<tool_response>{m['content']}</tool_response>"
        prev = r
    return s + "<|assistant|><think>"


def _harmony(msgs, keep="current"):
    ns = "\n\n".join(f"// {t['description']}\ntype {t['name']} = (_: {json.dumps(t['parameters'])}) => any;" for t in tools())
    s = ("<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n"
         "Knowledge cutoff: 2024-06\nCurrent date: 2026-09-24\n\nReasoning: medium\n\n"
         "# Valid channels: analysis, commentary, final. Channel must be included for every message.\n"
         "Calls to these tools must go to the commentary channel: 'functions'.<|end|>"
         f"<|start|>developer<|message|># Instructions\n\n{msgs[0]['content']}\n\n# Tools\n\n## functions\n\n"
         f"namespace functions {{\n\n{ns}\n\n}} // namespace functions<|end|>")
    last_call = None
    for i, m in enumerate(msgs[1:], 1):
        r = m["role"]
        if r == "user":
            s += f"<|start|>user<|message|>{m['content']}<|end|>"
        elif r == "assistant":
            if m.get("reasoning") and _keep_reasoning(msgs, i, keep):
                s += f"<|start|>assistant<|channel|>analysis<|message|>{m['reasoning']}<|end|>"
            calls = m.get("tool_calls") or ()
            if m["content"] and not calls:
                s += f"<|start|>assistant<|channel|>final<|message|>{m['content']}<|end|>"
            for c in calls:
                s += (f"<|start|>assistant<|channel|>commentary to=functions.{c['name']} <|constrain|>json"
                      f"<|message|>{json.dumps(c['arguments'])}<|call|>")
                last_call = c["name"]
        else:  # the real template renders tool output with `|tojson`
            s += f"<|start|>functions.{last_call} to=assistant<|channel|>commentary<|message|>{json.dumps(m['content'])}<|end|>"
    return s + "<|start|>assistant"


def _minimax(msgs, keep="all"):
    tj = "\n".join(f"<tool>{json.dumps(t, ensure_ascii=False)}</tool>" for t in tools())
    s = (f"]~!b[]~b]system\n{msgs[0]['content']}\n\n# Tools\nYou may call one or more tools to assist with the user query.\n"
         f"Here are the tools available in JSONSchema format:\n\n<tools>\n{tj}\n</tools>\n\n"
         f"When making tool calls, use XML format to invoke tools and pass parameters:\n\n<minimax:tool_call>\n"
         f"<invoke name=\"tool-name-1\">\n<parameter name=\"param-key-1\">param-value-1</parameter>\n</invoke>\n"
         f"</minimax:tool_call>[e~[\n")
    prev = None
    for i, m in enumerate(msgs[1:], 1):
        r = m["role"]
        if r == "user":
            s += f"]~b]user\n{m['content']}[e~[\n"
        elif r == "assistant":
            s += "]~b]ai\n"
            if m.get("reasoning") and _keep_reasoning(msgs, i, keep):
                s += f"<think>\n{m['reasoning']}\n</think>\n\n"
            s += m["content"]
            if m.get("tool_calls"):
                s += "\n<minimax:tool_call>\n"
                for c in m["tool_calls"]:
                    params = "".join(f"<parameter name=\"{k}\">{v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)}</parameter>\n"
                                     for k, v in c["arguments"].items())
                    s += f"<invoke name=\"{c['name']}\">\n{params}</invoke>\n"
                s += "</minimax:tool_call>"
            s += "[e~[\n"
        else:
            if prev != "tool":
                s += "]~b]tool"
            s += f"\n<response>{m['content']}</response>"
            if i == len(msgs) - 1 or msgs[i + 1]["role"] != "tool":
                s += "[e~[\n"
        prev = r
    return s + "]~b]ai\n<think>\n"


def _deepseek(msgs, keep="none"):
    tj = "\n\n".join(f"### {t['name']}\nDescription: {t['description']}\n\nParameters: {json.dumps(t['parameters'])}" for t in tools())
    s = (f"<｜begin▁of▁sentence｜>{msgs[0]['content']}\n\n## Tools\nYou have access to the following tools:\n\n{tj}\n\n"
         f"IMPORTANT: ALWAYS adhere to this exact format for tool use:\n<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>"
         f"tool_call_name<｜tool▁sep｜>tool_call_arguments<｜tool▁call▁end｜><｜tool▁calls▁end｜>")
    for m in msgs[1:]:
        r = m["role"]
        if r == "user":
            s += f"<｜User｜>{m['content']}"
        elif r == "assistant":
            s += f"<｜Assistant｜></think>{m['content']}"
            if m.get("tool_calls"):
                s += "<｜tool▁calls▁begin｜>" + "".join(
                    f"<｜tool▁call▁begin｜>{c['name']}<｜tool▁sep｜>{json.dumps(c['arguments'], ensure_ascii=False)}<｜tool▁call▁end｜>"
                    for c in m["tool_calls"]) + "<｜tool▁calls▁end｜>"
            s += "<｜end▁of▁sentence｜>"
        else:
            s += f"<｜tool▁output▁begin｜>{m['content']}<｜tool▁output▁end｜>"
    return s + "<｜Assistant｜></think>"


def _kimi(msgs, keep="none"):
    s = (f"<|im_system|>tool_declare<|im_middle|>{json.dumps([{'type': 'function', 'function': t} for t in tools()], ensure_ascii=False)}<|im_end|>"
         f"<|im_system|>system<|im_middle|>{msgs[0]['content']}<|im_end|>")
    for m in msgs[1:]:
        r = m["role"]
        if r == "user":
            s += f"<|im_user|>user<|im_middle|>{m['content']}<|im_end|>"
        elif r == "assistant":
            s += f"<|im_assistant|>assistant<|im_middle|>{m['content']}"
            if m.get("tool_calls"):
                s += "<|tool_calls_section_begin|>" + "".join(
                    f"<|tool_call_begin|>functions.{c['name']}:{c['id'].split('_')[1]}<|tool_call_argument_begin|>"
                    f"{json.dumps(c['arguments'], ensure_ascii=False)}<|tool_call_end|>" for c in m["tool_calls"]
                ) + "<|tool_calls_section_end|>"
            s += "<|im_end|>"
        else:
            s += (f"<|im_system|>tool<|im_middle|>## Return of functions.{m['name']}:{m['tool_call_id'].split('_')[1]}\n"
                  f"{m['content']}<|im_end|>")
    return s + "<|im_assistant|>assistant<|im_middle|>"


def _zephyr(msgs, keep="none"):
    """TinyLlama-Chat (Llama-2 SentencePiece / Metaspace vocab). Not an opencode
    model: included in the correctness check only, as the edge case for any
    cache that reuses segment encodings (Metaspace's `prepend_scheme: first`
    makes a segment's encoding depend on whether it starts the input)."""
    s = f"<|system|>\n{msgs[0]['content']}</s>\n"
    for m in msgs[1:]:
        r = m["role"]
        if r == "user":
            s += f"<|user|>\n{m['content']}</s>\n"
        elif r == "assistant":
            calls = "".join(f"\n{json.dumps(c)}" for c in m.get("tool_calls") or ())
            s += f"<|assistant|>\n{m['content']}{calls}</s>\n"
        else:
            s += f"<|user|>\nTool result:\n{m['content']}</s>\n"
    return s + "<|assistant|>\n"


# name -> (hub repo, renderer, kind). kind "hf" = tokenizer.json (reference:
# HF `tokenizers`); "tiktoken" = tiktoken.model (reference: `tiktoken`).
MODELS = {
    "qwen3-coder": ("Qwen/Qwen3-Coder-30B-A3B-Instruct", _qwen3_coder, "hf"),
    "glm-4.6": ("zai-org/GLM-4.6", _glm, "hf"),
    "glm-5.2": ("zai-org/GLM-5.2", _glm5, "hf"),
    "gpt-oss": ("openai/gpt-oss-120b", _harmony, "hf"),
    "minimax-m2": ("MiniMaxAI/MiniMax-M2", _minimax, "hf"),
    "deepseek-v3.1": ("deepseek-ai/DeepSeek-V3.1", _deepseek, "hf"),
    "kimi-k2": ("moonshotai/Kimi-K2-Instruct", _kimi, "tiktoken"),
    "tinyllama": ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", _zephyr, "hf"),
}
OPENCODE_MODELS = [m for m in MODELS if m != "tinyllama"]


def render(model: str, msgs) -> str:
    return MODELS[model][1](msgs)


# ── Tokenizer loaders ────────────────────────────────────────────────────────


def _hub(repo, filename):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, filename)


def _kimi_specials() -> tuple[dict[str, int], set[str]]:
    """Kimi reserves 256 ids after the ranks, names those its tokenizer_config
    declares and fills the rest with `<|reserved_token_{id}|>`. Returns the
    full table plus the declared-non-special names (the tool-call markers),
    which stay control tokens even when special tokens are split."""
    from tiktoken.load import load_tiktoken_bpe

    repo = MODELS["kimi-k2"][0]
    n = len(load_tiktoken_bpe(_hub(repo, "tiktoken.model")))
    with open(_hub(repo, "tokenizer_config.json"), encoding="utf-8") as f:
        declared = {int(k): v for k, v in json.load(f)["added_tokens_decoder"].items()}
    table = {declared[i]["content"] if i in declared else f"<|reserved_token_{i}|>": i for i in range(n, n + 256)}
    return table, {v["content"] for v in declared.values() if not v["special"]}


def load_fastokens(model: str, **kwargs):
    """The fastokens tokenizer as a server would build it. Honors the
    FASTOKENS_INPUT_CACHE / FASTOKENS_BPE_THREADS env vars set by the caller."""
    from fastokens._native import Tokenizer

    repo, _, kind = MODELS[model]
    if kind == "hf":
        return Tokenizer.from_file(_hub(repo, "tokenizer.json"), **kwargs)
    # `from_model` resolves the tiktoken-only layout (pattern, declared special
    # flags) the way a server would, but it probes the Hub for tokenizer.json on
    # every call, so a transient 5xx fails it: retry those.
    for attempt in range(4):
        try:
            return Tokenizer.from_model(repo, **kwargs)
        except ValueError as e:
            if "status code 5" not in str(e) or attempt == 3:
                raise
            time.sleep(2 ** attempt)


class Reference:
    """The reference implementation for `model`, behind one interface.

    encode_batch(texts)       special tokens recognized (chat-template output)
    encode_text_batch(texts)  special tokens encoded as plain text; tokens
                              declared non-special still match
    decode(ids, skip)         ids -> text
    """

    def __init__(self, model: str):
        repo, _, kind = MODELS[model]
        self.kind = kind
        if kind == "hf":
            from tokenizers import Tokenizer

            self.tok = Tokenizer.from_file(_hub(repo, "tokenizer.json"))
        else:
            import tiktoken
            from tiktoken.load import load_tiktoken_bpe

            sys.path.insert(0, os.path.join(REPO, "examples"))
            from validate_tiktoken import KIMI_PATTERN

            table, self.non_special = _kimi_specials()
            self.special_ids = {i for t, i in table.items() if t not in self.non_special}
            self.tok = tiktoken.Encoding(
                name="kimi-k2", pat_str=KIMI_PATTERN,
                mergeable_ranks=load_tiktoken_bpe(_hub(repo, "tiktoken.model")),
                special_tokens=table,
            )

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        if self.kind == "hf":
            return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]
        return self.tok.encode_batch(texts, allowed_special="all")

    def encode_text_batch(self, texts: list[str]) -> list[list[int]]:
        if self.kind == "hf":
            self.tok.encode_special_tokens = True
            try:
                return [e.ids for e in self.tok.encode_batch(texts, add_special_tokens=False)]
            finally:
                self.tok.encode_special_tokens = False
        return self.tok.encode_batch(texts, allowed_special=self.non_special, disallowed_special=())

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        if self.kind == "hf":
            return self.tok.decode(ids, skip_special_tokens=skip_special_tokens)
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_ids]
        return self.tok.decode(ids)


# ── Workload description ─────────────────────────────────────────────────────


def common_prefix(a: str, b: str) -> int:
    n = min(len(a), len(b))
    lo, hi = 0, n
    while lo < hi:  # binary search on slice equality (C-speed compares)
        mid = (lo + hi + 1) // 2
        if a[:mid] == b[:mid]:
            lo = mid
        else:
            hi = mid - 1
    return lo


def describe(model: str, seed: int = 0) -> str:
    """One-paragraph summary of a session's request stream for `model`."""
    prompts = [render(model, m) for m in session_requests(seed)]
    total = sum(len(p) for p in prompts)
    reused = sum(common_prefix(prompts[i - 1], prompts[i]) for i in range(1, len(prompts)))
    return (f"{model}: {len(prompts)} requests, largest {max(map(len, prompts)) / 1e3:.0f}K chars, "
            f"{total / 1e6:.1f}M chars tokenized in total; {reused / total:.0%} of those chars are a "
            f"byte-prefix repeat of the previous request")


if __name__ == "__main__":
    for name in sys.argv[1:] or OPENCODE_MODELS:
        print(describe(name))
