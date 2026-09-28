# fastokens 编译与使用说明（agent 负载优化）

本文说明如何从源码编译 fastokens、如何在服务中使用针对 coding agent（如 opencode）负载的几项优化，以及如何用配套的测试工具验证正确性和测量性能。

## 1. 改动一览

| 提交 | 内容 | 是否需要开关 | 效果（M2 Pro 实测） |
|---|---|---|---|
| `a4bccb7` `c6faa37` `9a9ac51` | agent 负载测试工具：正确性检查、性能基准、流量模拟（`examples/agent/`） | — | — |
| `293df0c` | 编码较长输入（≥16 KB）时释放 Python GIL；修复 `encode_batch_flat` 与 `enable_truncation` 等并发时会卡死整个进程的死锁 | 自动生效 | 编码期间其他线程被卡住的时间（p99）从 10–23 ms 降到约 1.6 ms；8 个线程的总吞吐提升 2–4 倍 |
| `20346fc` | 带特殊标记（chat template 输出）的请求也走手写的快速切词器 | 自动生效，仅限 o200k / Kimi 类切词规则（gpt-oss、Kimi 等） | 20 万 token 的请求：gpt-oss 8.8 → 1.2 ms，Kimi 40 → 2.0 ms |
| `bb81622` | 段缓存：按内容缓存特殊标记之间每一段文字的编码结果，多轮对话中只计算新出现的段 | 需开启 `FASTOKENS_SEGMENT_CACHE` | 一个 153 次请求的 agent 会话总耗时：Qwen3-Coder、GLM、MiniMax 快 5.7–6.8 倍，DeepSeek 快 15 倍 |
| `6813d6a` | 生成 Python 列表（`ids`、`attention_mask` 等）时复用整数对象、批量填充相同值 | 自动生效 | 经 transformers 调用时，开段缓存的单次请求再快 1.6–1.8 倍 |

以 GLM-5.2 为例，128 个 opencode 会话的真实流量模拟中，分词给每个请求增加的延迟（p50 / p99）：HF tokenizers 为 193 / 1716 ms，fastokens 改动前为 9.9 / 31 ms，改动后开启段缓存为 4.7 / 10.9 ms。

所有改动都保证结果与原实现逐 token 一致，验证方式见第 4 节。

## 2. 编译

### 2.1 安装 Rust

需要 Rust stable 工具链。仓库代码用到了 let-chains 语法，要求 Rust 1.88 及以上。用 rustup 安装：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --component rustfmt,clippy
```

按上面的命令安装时，rustup 会把 PATH 写入 shell 配置文件，新开的终端会自动生效。如果安装时加了 `--no-modify-path`（本机就是这样安装的），每个新终端都要先执行：

```bash
source "$HOME/.cargo/env"
```

PCRE2 会被静态编译进去（见 `.cargo/config.toml` 中的 `PCRE2_SYS_STATIC=1`），不需要安装系统依赖。

### 2.2 编译并测试 Rust 库

仓库根目录就是 Rust 库本身（`src/`）：

```bash
cargo build --release
```

运行测试。部分测试第一次运行时会从 Hugging Face 下载分词器文件，偶尔会因并发下载抢锁或网络 5xx 失败，重跑即可：

```bash
cargo test --release
```

提交代码前，跑 CI 使用的两项检查：

```bash
cargo clippy -- -D warnings
```

```bash
cargo fmt --check
```

### 2.3 编译 Python 扩展

Python 包由 [maturin](https://github.com/PyO3/maturin) 构建：`pyproject.toml` 指向 `python/Cargo.toml`，生成的模块是 `fastokens._native`。

**推荐方式：编译并直接安装到虚拟环境。**

```bash
uv venv .venv --python 3.12
```

```bash
source .venv/bin/activate
```

```bash
uv pip install maturin pytest
```

```bash
maturin develop --release
```

**另一种方式：打成 wheel 再安装。** 适合要装到多个虚拟环境，或拷贝到其他机器的情况：

```bash
maturin build --release --out dist
```

```bash
uv pip install --reinstall dist/fastokens-*.whl
```

注意事项：

- `--release` 不能省。调试版比发布版慢很多，性能测试结果没有意义。
- 不能直接用 `cargo build -p fastokens-python` 编译 Python 扩展：在 macOS 上会链接失败，因为扩展里的 Python 符号要在加载时才解析，需要 maturin 传入对应的链接参数。
- 如果要交替运行 `cargo test` 和 maturin，可以给 maturin 设置单独的编译目录，避免两者互相覆盖、导致每次都整体重编：

```bash
CARGO_TARGET_DIR=target/py maturin develop --release
```

### 2.4 确认装上的是本地编译的版本

```bash
python -c "import fastokens; print(fastokens.__file__)"
```

输出的路径应当位于你的虚拟环境中。然后运行 Python 测试：

```bash
pytest python/tests
```

## 3. 使用

### 3.1 直接调用

```python
from fastokens import Tokenizer

tok = Tokenizer.from_model("zai-org/GLM-5.2")      # 或 Tokenizer.from_file("tokenizer.json")
ids = tok.encode(prompt).ids                       # prompt 为 chat template 渲染后的完整字符串
```

### 3.2 通过 transformers 使用（SGLang、vLLM 等服务框架）

服务框架通过 transformers 调用分词器。在创建分词器之前调用一次 `patch_transformers()`，之后 `AutoTokenizer` 就会使用 fastokens：

```python
import fastokens
fastokens.patch_transformers()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2")
```

服务进程需要在同一个进程里先打补丁、再启动服务。仓库中的 SGLang 示例（`examples/sglang_speed.py`）是这样启动的：

```bash
FASTOKENS_SEGMENT_CACHE=1024 python -c "import fastokens; fastokens.patch_transformers(); import runpy; runpy.run_module('sglang.launch_server', run_name='__main__')" --model-path zai-org/GLM-5.2
```

说明：

- 补丁只作用于调用了 `patch_transformers()` 的那个进程。服务框架里负责分词的是 API / tokenizer 进程，也就是上面启动的主进程。
- vLLM 可以用同样的 `runpy` 方式启动它的 OpenAI 服务模块。本次改动没有在 SGLang 或 vLLM 的真实部署中实测，上线前建议先用你的部署方式验证一遍。
- NVIDIA Dynamo 通过 `--tokenizer fastokens` 直接调用 Rust 库，不需要打补丁。要用上这些改动，需要用包含它们的版本编译 Dynamo：快速切词器自动生效，段缓存同样用环境变量开启；释放 GIL 和列表转换两项只作用于 Python 调用。

### 3.3 开关与参数

| 环境变量 / 接口 | 作用 | 默认 |
|---|---|---|
| `FASTOKENS_SEGMENT_CACHE=<MiB>` | 开启段缓存，数值是内存上限（缓存的文本加 token id）。Rust 中对应 `Tokenizer::enable_segment_cache(bytes)` | 关闭 |
| `FASTOKENS_INPUT_CACHE=<条数>` | 原有的前缀缓存。只对不含特殊标记的纯文本、且切词规则为 o200k / Kimi 类的输入有效，对 chat template 输出不起作用 | 关闭 |
| `FASTOKENS_BPE_THREADS=<数量>` | 单个请求内部并行使用的线程数 | 全部逻辑核（Apple Silicon 上为性能核数） |
| 释放 GIL | `encode`、`encode_ordinary`、`encode_segments`、`encode_batch` 在输入 ≥16 KB 时自动释放 | 始终开启 |
| 带标记请求走快速切词器 | 对 o200k / Kimi 类切词规则自动生效 | 始终开启 |
| Python 列表转换优化 | 自动生效。token id 对应的整数对象会常驻内存，按常见词表大小约占几 MB | 始终开启 |

环境变量在创建分词器时读取，所以必须在进程启动前、或调用 `patch_transformers()` 之前设置好。

### 3.4 段缓存怎么用

- **适用场景**：多轮对话、coding agent 这类每次请求都重发完整历史的负载。它按"特殊标记之间的一段文字"缓存，只对包含特殊标记的输入生效（chat template 的输出都属于这种情况），纯文本输入不受影响。
- **正确性**：每次命中都会和缓存的原文逐字节比对，结果与不开缓存完全一致。模板改写历史（例如删掉旧的思考内容）或触发 compact 时，没变的段依然能命中。
- **内存估算**：一个约 20 万 token 的 agent 会话约占 1.5 MiB，多个会话共享的部分（系统提示、工具定义）只存一份。建议值：几十个并发会话设 256，上百个会话设 1024。超过上限时，自动淘汰最久没用到的内容。
- **何时不必开**：请求之间几乎没有重复内容（如一次性的长文档处理）时，开启只会增加查表开销，收益为零。

## 4. 验证与测量

配套工具在 `examples/agent/` 中，详细说明见 [`examples/agent/README.md`](../examples/agent/README.md)。它们需要额外安装：

```bash
uv pip install huggingface_hub tokenizers tiktoken "transformers<5"
```

**正确性检查**：把模拟的 agent 请求交给 fastokens 的各条调用路径，要求结果与标准实现（HF tokenizers，Kimi 为 tiktoken）逐 token 一致：

```bash
python examples/agent/check.py --quick
```

```bash
python examples/agent/check.py --transformers
```

**性能基准**：覆盖单请求、整个会话、多会话交错、GIL 卡顿、多线程吞吐等场景，可以保存结果并和上一次对比：

```bash
python examples/agent/bench.py --json base.json
```

```bash
python examples/agent/bench.py --json new.json --compare base.json
```

**真实流量模拟**：模拟少量或大量 opencode 会话持续增长、偶尔 compact 的请求流，经 transformers 调用分词器：

```bash
FASTOKENS_SEGMENT_CACHE=1024 python examples/agent/simulate.py --impl fastokens --model glm-5.2 --sessions 128 --json ft.json
```

```bash
python examples/agent/simulate.py --impl hf --model glm-5.2 --sessions 128 --json hf.json
```

**对比两个版本**：用 `git worktree` 把旧版本检出到单独目录，编译后装进单独的虚拟环境，再用各自的 Python 解释器运行同一个测试脚本。当前工作目录不受影响。以编译 main 为例：

```bash
git worktree add ../fastokens-main main
```

```bash
cd ../fastokens-main && CARGO_TARGET_DIR=target/py-main maturin build --release --out dist-main
```

把 `dist-main` 里的 wheel 装进另一个虚拟环境后，就可以删掉这个检出目录：

```bash
git worktree remove ../fastokens-main
```

注意事项：

- 只在同一台空闲的机器上比较结果。
- 两个版本最好交替运行。长时间测试中，机器的发热等状态会让数字整体漂移。
- 同一份代码重复运行，汇总类指标的差异在 ±5% 以内；`--compare` 只标出超过 10% 的变化。

## 5. 已知问题（与本次改动无关）

- `python/tests/test_decode_sanitize.py` 中有 4 个测试在 main 上就失败（测试里构造的 BPE 配置解析失败）。CI 不运行这个文件，所以一直没被发现。
- `Tokenizer.from_model` 加载 Kimi 这类只有 `tiktoken.model` 的仓库时，每次都会联网查询一个并不存在的 `tokenizer.json`，网络抖动（如 HTTP 504）会导致加载失败，离线环境也无法使用。
- `cargo clippy --all-targets` 在 `examples/simple_bench.rs` 和 `src/models/bpe.rs` 的测试代码中有 2 条原有告警。CI 只检查 `cargo clippy`，不受影响。
- 文中所有性能数字都在 Apple M2 Pro（6 个性能核）上测得。服务器 CPU 单核通常更慢、核数更多，绝对数值会不同，但各版本之间的相对差异不变。
