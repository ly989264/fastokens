# Agent workload harness

Correctness and performance harness for **coding-agent traffic** (opencode and
similar), the workload where a tokenizer sees:

- **chat-templated prompts** full of special tokens (`<|im_start|>`,
  `<tool_call>`, `<|start|>…<|message|>`, …) rather than plain text;
- **the whole history on every request**: the chat API is stateless, so each
  agent step re-sends everything before it (~97% of each request's bytes
  repeat the previous request);
- **history rewrites** (templates drop old reasoning; auto-compaction replaces
  the history with a summary);
- **many sessions at once**, interleaved.

`workload.py` generates that stream deterministically (file contents come from
this repo at a pinned commit, so runs stay comparable while `src/` changes) and
renders it with each model's own special tokens:

| name | model | reference |
|---|---|---|
| `qwen3-coder` | Qwen/Qwen3-Coder-30B-A3B-Instruct | HF `tokenizers` |
| `glm-4.6` | zai-org/GLM-4.6 | HF `tokenizers` |
| `glm-5.2` | zai-org/GLM-5.2 | HF `tokenizers` |
| `gpt-oss` | openai/gpt-oss-120b | HF `tokenizers` |
| `minimax-m2` | MiniMaxAI/MiniMax-M2 | HF `tokenizers` |
| `deepseek-v3.1` | deepseek-ai/DeepSeek-V3.1 | HF `tokenizers` |
| `kimi-k2` | moonshotai/Kimi-K2-Instruct | `tiktoken` |
| `tinyllama` | TinyLlama/TinyLlama-1.1B-Chat-v1.0 | HF `tokenizers` (correctness only: Metaspace edge case) |

## Setup

```
pip install maturin huggingface_hub tokenizers tiktoken transformers
maturin develop --release        # build the extension from this checkout
```

Tokenizer files are downloaded from the Hugging Face Hub on first use.

## Correctness: `check.py`

Replays interleaved sessions through `encode`, `encode` with the prefix cache
on, with the segment cache on (roomy, and tiny so it evicts constantly),
`split_special_tokens=True`, `encode_ordinary` with the prefix cache on,
`encode_batch`, several threads sharing one tokenizer (with either cache),
`decode`, and optionally the patched `transformers` path. Every result must be bit-identical to the
reference; the first mismatch is printed with its context.

```
python examples/agent/check.py --quick           # ~1 min
python examples/agent/check.py --transformers    # full: 3 interleaved sessions per model
```

## Performance: `bench.py`

| scenario | measures |
|---|---|
| `single` | one request at 16k/64k/128k/200k tokens; `tmpl` (real path) vs `ord` (same bytes, no special-token split) |
| `session` | one agent session end to end: total / p50 / p99 encode time with no cache, the prefix cache and the segment cache |
| `server` | 16 sessions interleaved round-robin, with each cache variant |
| `gil` | worst stall of a 1 ms ticker thread while another thread encodes |
| `threads` | aggregate throughput with 1/2/4/8 Python client threads |
| `tf` | end-to-end through the patched `transformers` tokenizer |

A/B workflow:

```
git stash; maturin develop --release
python examples/agent/bench.py --json base.json
git stash pop; maturin develop --release
python examples/agent/bench.py --json new.json --compare base.json
```

Every timed call sees a distinct prompt, and timings are medians over seeds, so
whole-input caches cannot flatter a result. Compare only runs from the same,
otherwise idle machine. Re-running unchanged code moves aggregate metrics by
under 5% and sub-millisecond single requests by up to ~15%, so `--compare`
flags only changes beyond 10% and ignores the (single-sample) `max_gap_ms`.

## Traffic simulation: `simulate.py`

Replays realistic opencode traffic against a vLLM-style frontend: N agent
sessions (few or many), each a loop of whole-history requests with model/tool
time between them (log-normal, median 3 s) and longer user turns, growing
until auto-compaction, starting at random points of their lives (steady
state). One asyncio event loop renders requests and hands tokenization to a
thread pool through `transformers`; a 5 ms ticker stands in for the SSE
streaming of every other session. It reports the latency tokenization adds
per request (queueing + encode), event-loop lag and CPU per request.

```
python examples/agent/simulate.py --impl hf --sessions 128 --json hf.json
python examples/agent/simulate.py --impl fastokens --sessions 128 --json ft.json
FASTOKENS_SEGMENT_CACHE=1024 python examples/agent/simulate.py --impl fastokens --sessions 128
```

Each run measures whichever `tokenizers` / `fastokens` its interpreter
imports; compare builds with one virtualenv per build, run back to back.
