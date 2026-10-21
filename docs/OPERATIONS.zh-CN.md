# fastokens 操作说明：段缓存、前缀缓存和分析功能

适用版本：本仓库 main（`1a0f7e2`，在上游 v0.3.2 之上加了本 fork 的改动）。文中的命令、环境变量和接口都对着这个版本的代码核对过；哪些在本机实际跑过、哪些引用 CI 的结果，见第 9 节。

三样东西：

- **段缓存**（segment cache）：按内容缓存"两个特殊标记之间那一段文字"的编码结果。多轮对话每次都重发完整历史，旧的段直接命中，只编码新出现的段。
- **前缀缓存**（prefix cache）：找出和新请求共享最长开头的旧请求，复用到共享部分里最后一个特殊标记为止，只编码剩下的部分。
- **分析功能**（`fastokens.observe`）：看时间花在哪个阶段、缓存命中了多少、CPU 花在哪些函数上。

两种缓存都默认关闭，打开后结果与不开逐 token 一致。

## 0. 快速开始（A3 机器，经 vLLM 使用）

1. 在 GitHub 上触发 A3 构建（只读取 main 的代码，不会改动 main）：

```bash
gh workflow run build-a3.yml --repo ly989264/fastokens --ref main
```

2. 找到这次运行的编号（ID 一列），等它跑完（约 3 分钟）：

```bash
gh run list --repo ly989264/fastokens --workflow build-a3.yml --limit 1
```

3. 下载观测版（普通版把 `observe` 换成 `default`），装进 vLLM 所在的 Python 环境（A3 机器上没有 `gh` 时，在别的机器上下载后把 wheel 拷过去）：

```bash
gh run download <run-id> --repo ly989264/fastokens -p 'fastokens-a3-observe-*' -D dist-a3
```

```bash
python -m pip install --force-reinstall dist-a3/fastokens-a3-observe-*/fastokens-*.whl
```

4. 确认装上的是哪个变体：

```bash
python -c "import fastokens, fastokens.observe as o; print(fastokens.__file__); print('observe' if o.enabled() else 'default')"
```

5. 开两种缓存启动 vLLM（数值的定法见第 2–4 节）：

```bash
VLLM_USE_FASTOKENS=1 FASTOKENS_SEGMENT_CACHE=256 FASTOKENS_INPUT_CACHE=64 FASTOKENS_INPUT_CACHE_MIB=1024 vllm serve <模型> --api-server-count 4
```

这组值占多少内存（默认 1 个渲染线程，即每个 API 进程 2 份分词器拷贝；4 个 API 进程；式子见 2.4、3.2 节）：

- 段缓存最坏约 256 MiB × 2 × 4 × 1.25 ≈ 2.5 GiB；
- 前缀缓存最多 1024 MiB × 2 × 4 = 8 GiB。这是字节预算封顶的上限，实际用量通常小得多：每条约为请求文本的 2.2 倍，10 万 token 的会话一条约 0.8 MiB，64 条约 51 MiB 每份拷贝。
- `FASTOKENS_INPUT_CACHE=64` 按"条数取活跃会话数的 1.5–2 倍"算，约够每个 API 进程 32–42 个活跃会话；会话更多时按 3.2 节调大条数，再按内存调 `FASTOKENS_INPUT_CACHE_MIB`。

6. 确认生效：装上第 6.2 节的小插件，启动时再加 `FASTOKENS_OBSERVE_PORT=9464`，有流量之后看命中率。端口换成日志里 API 进程拿到的那个；这条命令要观测版，普通版没有命中计数，会报 `KeyError`：

```bash
curl -s http://127.0.0.1:9464/json | python -c "import json,sys; d=json.load(sys.stdin); [print(n, 'segment hit bytes', t['segment_cache']['hit_bytes'], 'miss bytes', t['segment_cache']['miss_bytes'], '| prefix exact/prefix/miss', t['prefix_cache']['chat']['exact_hits'], t['prefix_cache']['chat']['prefix_hits'], t['prefix_cache']['chat']['misses']) for n, t in d['tokenizers'].items()]"
```

## 1. 拿到 wheel

### 1.1 两个变体怎么选

| | 普通版（default） | 观测版（observe） |
|---|---|---|
| 编译参数 | 无 | `--features observe` |
| 编码结果、两种缓存 | 一样 | 一样 |
| `fastokens.observe.enabled()` | `False` | `True` |
| 缓存能看到什么 | 只有条数和占用字节 | 加上命中/未命中次数和字节数、淘汰、锁等待等 |
| 阶段耗时、路径、最慢请求、Python 接口的 Rust / 等 GIL / Python 拆分 | 没有 | 有 |
| `serve` / `dump` | 能用，只有缓存大小等基础数据 | 能用 |
| `top` / `profile` / `report()` | 只打印 "built without the `observe` feature: nothing to show" | 能用 |
| `sample`（CPU 采样） | 能用 | 能用 |
| `layers`（分层拆分） | 不能用（报错） | 能用 |
| 适合 | 正式部署 | 要确认缓存生效、查时间花在哪 |

观测版的开销测不出来：在同一个 commit 上只差 `--features observe`，用 GLM-5.3 agent 流量（1920 个请求，开关缓存 × 1 / 4 个线程，M2 Pro，perflock 下交替 5 轮、逐轮配对）比较，差别在 −0.7% 到 +1.5% 之间（正数表示观测版更快），属于测量波动（main `508cbe4`，见 [`OBSERVE.zh-CN.md`](OBSERVE.zh-CN.md)）。所以可以直接在线上用观测版。

### 1.2 普通版：每次推到 main 自动编

`Build and Release fastokens` 在推到 main、且改动涉及 `python/`、`src/`、`Cargo.toml`、`Cargo.lock` 或 `.github/workflows/` 时自动运行（只改文档不运行），编出各平台的普通版 wheel，名字是 `wheels-<系统>-<架构>`。A3 用 `wheels-linux-aarch64`（manylinux_2_28，Python 3.9 及以上通用）。

```bash
gh run list --repo ly989264/fastokens --workflow build-release-fastokens.yml --branch main --limit 5
```

```bash
gh run download <run-id> --repo ly989264/fastokens -n wheels-linux-aarch64 -D dist
```

```bash
python -m pip install --force-reinstall dist/fastokens-*.whl
```

### 1.3 A3 的两个变体：手动触发

`Build fastokens for A3`（`.github/workflows/build-a3.yml`）只能手动触发，一次编出普通版和观测版两个 aarch64 wheel，并在 GitHub 的 arm64 机器上各跑一遍冒烟测试（两种缓存的开关组合逐 token 一致、观测版的分层拆分和快照、Linux `perf` 采样）。命令见第 0 节第 1–3 步。

- 触发只是让 GitHub 读取这个分支的代码来编译，不会改动 main，也不会发布到 PyPI；编其他分支把 `--ref` 换成分支名。
- 两个 artifact 的名字是 `fastokens-a3-default-<提交号>` 和 `fastokens-a3-observe-<提交号>`（完整 40 位），用 `-p 'fastokens-a3-default-*'` 或 `-p 'fastokens-a3-observe-*'` 下载，wheel 在 `dist-a3/fastokens-a3-<变体>-<提交号>/` 里。
- artifact 保留 90 天。

### 1.4 本机自己编

需要 Rust 1.88+ 和 maturin（详见 [`BUILD_AND_USAGE.zh-CN.md`](BUILD_AND_USAGE.zh-CN.md) 第 2 节）。普通版：

```bash
maturin build --release --out dist
```

观测版（两个变体要装进不同的虚拟环境；在同一台机器上交替编时，给它们不同的 `CARGO_TARGET_DIR`）：

```bash
CARGO_TARGET_DIR=target/observe maturin build --release --features observe --out dist-observe
```

### 1.5 确认装的是哪个变体

```bash
python -c "import fastokens, fastokens.observe as o; print(fastokens.__file__); print('observe' if o.enabled() else 'default')"
```

第一行的路径应当在你要用的 Python 环境里；如果环境里还装着别的 fastokens（例如 PyPI 的上游版），以这里打印的为准。

## 2. 段缓存

### 2.1 原理

特殊标记（`<|user|>`、`<|assistant|>`、`<tool_call>` 等）是分词的硬边界：两个标记之间的那段文字怎么切分、编出什么 id，只取决于这段文字本身。所以可以按内容缓存每一段的结果，命中时逐字节比对原文后直接复用。agent 会话每次请求都带着完整历史，绝大部分段都是旧的。

### 2.2 开启

```bash
export FASTOKENS_SEGMENT_CACHE=256
```

- 数值是内存预算，单位 MiB；`0`、不设或不是数字都表示关闭。
- **只能用环境变量开**：main 上 Python 的 `Tokenizer` 没有开关方法（Rust 里是 `Tokenizer::enable_segment_cache(bytes)`）。
- 变量在**创建分词器时**读取，所以要在进程启动前设好；已经创建的分词器不受影响。
- 每个分词器实例各有一份缓存，预算按实例计。
- 只对含特殊标记的输入生效（chat template 的输出都是），纯文本输入不受影响。
- 成本超过预算 1/64 的单段不缓存（成本 = 文字字节数 + 每个 token 4 字节 + 64 字节），照常编码。

### 2.3 预算怎么定

在 main 上用 GLM-5.3 模拟 opencode 会话做过容量扫描（直接调用 fastokens 编码，observe 构建读命中计数，5 档并发会话 × 12 档预算，前 1/3 流量当热身；main `b7d0d42`，代码与 `1a0f7e2` 相同）：

| 并发会话 | 实际工作集 | 字节命中率第一次 ≥95% 的预算 |
|---|---|---|
| 1 | 0.6 MiB | 8 MiB（95.6%） |
| 4 | 2.0 MiB | 8 MiB（95.2%） |
| 16 | 8.5 MiB | 32 MiB（98.6%；16 MiB 是 94.0%） |
| 64 | 26.6 MiB | 64 MiB（98.0%） |
| 256 | 109.3 MiB | 256 MiB（98.4%） |

由此得出两条规则：

- **规则 A**：预算取工作集的 2.3–2.5 倍（负载大时成立）。粗估时，一个约 20 万 token 的 agent 会话约 1.5 MiB；会话之间共享的系统提示、工具定义只存一份。
- **规则 B**：预算至少是最大单段成本的 64 倍，否则最大的段存不进去。这批流量里最大的段是 133,165 字节，所以至少 8.5 MiB，1 个和 4 个会话也要 8 MiB 才到 95%。

命中率封顶之后，总编码时间是不开缓存的 0.22–0.23 倍（同一次扫描）。建议起步值：几十个并发会话设 256，上百个设 1024，再用第 2.6 节的命中率核对。

### 2.4 在 vLLM 里

vLLM 0.23 自带开关 `VLLM_USE_FASTOKENS=1`：它在创建分词器时调用 `fastokens.patch_transformers()`（`vllm/tokenizers/registry.py`），之后 transformers 的分词器都走 fastokens。

```bash
VLLM_USE_FASTOKENS=1 FASTOKENS_SEGMENT_CACHE=256 vllm serve <模型> --api-server-count 4
```

- **变量要进每个 API 进程**：`--api-server-count N` 会再启动 N 个 API 进程，它们继承 `vllm serve` 的环境变量。所以写在启动命令前、或容器 / systemd 的环境里即可；不要只在某个子 shell 里设。多台机器每台都要设。
- **拷贝池**：每个 API 进程把分词器深拷贝成 `--renderer-num-workers` + 1 份（默认 1 + 1 = 2 份，`vllm/renderers/hf.py`），同时借用的请求比拷贝多时还会临时再拷一份并留在池里。每份拷贝有自己的段缓存，所以：

  实际内存上限 ≈ 预算 × (`--renderer-num-workers` + 1) × `--api-server-count` × 1.25（1.25 见 2.7）

  例如预算 256 MiB、默认 1 个渲染线程、4 个 API 进程：256 × 2 × 4 = 2 GiB，最坏约 2.5 GiB。第 6.2 节的插件会列出每个进程实际有几份分词器。
- **会话亲和**：缓存每个进程一份，同一个会话的请求被随机分到不同进程时，每个进程都要重新编码它的段。实测（vLLM 0.30 的真实链路，GLM-5.3 模拟 opencode 会话，main `508cbe4`，M2 Pro）：随机分配比按会话分配 CPU 多 8–9%，命中率从 97% 降到 90%。vLLM 自己不做亲和，要在网关 / 负载均衡上按会话 id 路由到同一个 vLLM 实例；同一实例内部的多个 API 进程之间 vLLM 不按会话分配，所以这部分损失目前避免不了。
- **多进程好于多线程**：开缓存时加 API 进程有效（4 个进程吞吐 3.7–3.8 倍），加 `--renderer-num-workers` 没用（受 GIL 限制）。拷贝池相对"共用一个分词器"的代价主要是内存（4 个渲染线程多 490–680 MB），吞吐只差 −0.5% 到 −1.6%（同一组实测）。

### 2.5 直接在 Python 里用

```python
import os
os.environ["FASTOKENS_SEGMENT_CACHE"] = "256"   # 必须在创建分词器之前

import fastokens
fastokens.patch_transformers()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("zai-org/GLM-5.2")
```

或者不经过 transformers：`fastokens.Tokenizer.from_file("tokenizer.json")`，同样在创建前设好变量。

### 2.6 怎么确认生效

缓存的状态在每个分词器上：

```python
import json, fastokens.observe as obs
snap = obs.snapshot({"glm": tok})            # tok 可以是 transformers 分词器、shim 或原生 Tokenizer
print(json.dumps(snap["tokenizers"]["glm"]["segment_cache"], indent=1))
```

- **观测版**：有 `hits`、`misses`、`hit_bytes`、`miss_bytes`。字节命中率 = `hit_bytes / (hit_bytes + miss_bytes)`，稳定的多轮负载上应在 90% 以上；还有 `rejected_too_large`（因超过 1/64 没存的段）和每个分片的写入、淘汰、锁等待。
- **普通版**：只有 `entries`、`bytes`、`approx_max_bytes` 等大小信息，**看不到命中率**。`entries` 大于 0 只能说明缓存开着、在写入。
- vLLM 里用第 6.2 节的插件看，或第 0 节第 6 步的一行命令。

### 2.7 已知问题

- **预算不是硬上限**：main 上先写入、再轮换，理论上最多到预算的 1.25 倍。实测最高 1.094 倍，进程内存的增加与缓存自报的字节数一致（1.00–1.02 倍）（main `508cbe4`）。按 1.25 倍留内存余量。
- 改词表（`add_tokens`）或 normalizer（`set_normalizer`）会清空缓存。
- 带 BPE dropout 的模型结果本身是随机的，开缓存会把第一次的结果固定下来；常见推理模型都不带 dropout。

### 2.8 关闭和回滚

- 关闭：去掉 `FASTOKENS_SEGMENT_CACHE`（或设为 `0`），重启进程。
- 不用 fastokens：vLLM 里去掉 `VLLM_USE_FASTOKENS=1`；或卸载本 fork 的 wheel，装回原来的版本。

## 3. 前缀缓存

### 3.1 原理和开启

```bash
export FASTOKENS_INPUT_CACHE=64 FASTOKENS_INPUT_CACHE_MIB=1024
```

- `FASTOKENS_INPUT_CACHE=<条数>`：保留最近的多少个请求；`0`、不设或不是数字表示关闭。
- `FASTOKENS_INPUT_CACHE_MIB=<MiB>`：字节预算，默认 **2048**；条数或字节数任一超限就淘汰最久没用的条目。每条约占请求文本的 2.2 倍（文本加每 token 4 字节）。成本超过预算 1/64 的请求不存。
- 同样只能用环境变量开，在创建分词器时读取，每个实例各一份。
- 含特殊标记的输入（chat template 的输出）：找共享开头最长的旧请求，复用到共享部分里最后一个安全切点（特殊标记的起点），只编码后面的部分；段缓存也开着时，后面这部分再走段缓存。短于 4 KiB 的输入不查找。
- 新请求延续了被复用的那个会话时替换旧条目；只和它共享系统提示的其他会话不互相顶掉。
- 不含特殊标记的纯文本：沿用原来的逻辑（只对 o200k / Kimi 类切词规则生效），只受条数限制。

### 3.2 条数和预算怎么定

- **条数 ≥ 活跃会话数，建议取 1.5–2 倍**。每个活跃会话需要一条（它最新的那个请求）；条数不够时，一个会话的条目常在它下一次请求之前就被淘汰，这个请求几乎要整段重编。另外"用户改写了大部分历史"之后旧版本不会被替换，会占着槽位直到淘汰，所以要留余量。
- **字节预算**：约等于 条数 × 单个请求的大小 × 2.2。例如 10 万 token 的会话（约 40 万字符），一条约 0.8 MiB，`FASTOKENS_INPUT_CACHE=512` 约占 400 MiB 左右；请求长到 100 万 token 就是十倍，所以要设 `FASTOKENS_INPUT_CACHE_MIB` 封顶。
- **在 vLLM 里**：和段缓存一样，每个 API 进程的每份分词器拷贝各一份，变量要进每个 API 进程。同一个会话的相邻请求可能落到不同的拷贝上，具体分配不固定；按最坏情况定，**每份拷贝的条数都要按"能见到全部活跃会话"来定**（每个进程只算分给它的会话）。

  实际内存上限 ≈ min(条数 × 单条大小, 字节预算) × (`--renderer-num-workers` + 1) × `--api-server-count`

  默认预算 2048 MiB 是按每份拷贝算的，多进程、多拷贝时务必按上式核对，必要时调小 `FASTOKENS_INPUT_CACHE_MIB`。

### 3.3 效果（main 实测）

GLM-5.2，256 个会话 × 4000 个请求的回放（`examples/agent/prefix_bench.py`，直接调用 fastokens，M2 Pro，5 个种子；基线 main `508cbe4`，最终代码复测数字一致，详见 [`PREFIX_CACHE_INFERENCE.zh-CN.md`](PREFIX_CACHE_INFERENCE.zh-CN.md) 第 3 节）：

| | 都不开 | 只开段缓存 | 只开前缀缓存 | 两个都开 |
|---|---|---|---|---|
| 总耗时比都不开快 | 1× | 4.26–4.43× | 4.74–4.93× | **5.29–5.52×** |
| 每请求 CPU | 9.05–10.11 ms | 1.14–1.34 ms | 1.10–1.27 ms | 0.92–1.08 ms |
| 单请求 p99 | 10.62 ms | 2.62 ms | 3.22 ms | 2.12 ms |

- 历史追加的请求（91%）上前缀缓存比段缓存快 16–17%；"用户新发言、历史被改写"的请求上前缀缓存只能复用到改写点，耗时是段缓存的 1.7 倍左右。两个都开没有这个短板。
- 前缀缓存要比段缓存多用约 1.8 倍内存才能快这么多：它按请求存，段缓存按内容去重。

### 3.4 怎么确认生效

```python
print(json.dumps(obs.snapshot({"glm": tok})["tokenizers"]["glm"]["prefix_cache"]["chat"], indent=1))
```

- **观测版**：`lookups`（查找次数）、`exact_hits`（完全相同的请求）、`prefix_hits`（复用了开头）、`misses`、`reused_bytes` / `encoded_bytes`（复用和重新编码的字节数）、`cut_backoffs`、`no_cut`、`bypass_short`（短于 4 KiB 没查）、`inserts` / `replaced` / `evicted_entries`、`rejected_oversized`。复用比例 = `reused_bytes / lookup_bytes`。
- **普通版**：只有 `entries`、`bytes`、`capacity`、`max_bytes`、`index_blocks`，看不到命中率。

### 3.5 已知问题

- 条数少于活跃会话数时几乎不起作用（见 3.2）。
- 短于 4 KiB 的请求不查找；ShareGPT 这类短对话里接近一半的请求属于这种，只有段缓存能帮上。
- 改写历史的请求只复用到改写点（两种缓存一起开就没问题）。
- main 原有的特殊标记切分问题（与缓存无关，开不开都一样）：一个带 `rstrip` 的特殊标记吞掉的空白里开始一个带 `lstrip` 的标记时会 panic。测过的 16 个真实模型都没有这种标记。
- 关闭和回滚同 2.8（去掉 `FASTOKENS_INPUT_CACHE`）。

## 4. 两个都开（推荐）

```bash
VLLM_USE_FASTOKENS=1 FASTOKENS_SEGMENT_CACHE=256 FASTOKENS_INPUT_CACHE=64 FASTOKENS_INPUT_CACHE_MIB=1024 vllm serve <模型> --api-server-count 4
```

这组值的内存估算见第 0 节第 5 步。

- 前缀缓存先复用开头，剩下的部分再走段缓存；改写历史的请求由段缓存兜底。上面的回放里两个都开最快（比只开段缓存少用 19–20% 的时间，比只开前缀缓存少用 10–12%）。
- 两个都开时，单个分词器实例最多占两者之和（段缓存预算 × 1.25 + 前缀缓存字节预算），再乘以拷贝份数和 API 进程数。
- 定数值的顺序：
  1. 按会话亲和后每个 API 进程的活跃会话数定前缀缓存条数（1.5–2 倍）；
  2. 按请求大小定 `FASTOKENS_INPUT_CACHE_MIB`；
  3. 按 2.3 节定段缓存预算；
  4. 按上面的式子核对总内存；
  5. 上线后用观测版看两个缓存的命中率（第 6.2 节），按需调整。

## 5. 分析功能

| 工具 | 做什么 | 需要的变体 | 用法 |
|---|---|---|---|
| `serve` | 在进程里开一个 HTTP 服务：仪表盘 `/`、快照 `/json`、Prometheus `/metrics` | 任意（普通版只有缓存大小） | Python：`observe.serve(...)` |
| `top` | 终端里实时看一个已开 `serve` 的进程 | **观测版** | `python -m fastokens.observe top` |
| `profile` | 对一个已开 `serve` 的进程采一段时间，打印分析 | **观测版** | `python -m fastokens.observe profile`；或 Python `with observe.profile(...)` |
| `dump` | 打印一次 `/json` 快照 | 任意（普通版只有缓存大小） | `python -m fastokens.observe dump` |
| `analyze` / `report` | 两个快照之差：分析数据 / 文字报告 | **观测版** | Python：`observe.analyze(prev, cur)`、`observe.report(prev, cur)` |
| `sample` | CPU 采样，把每个样本归到流水线阶段、分配器、正则引擎、GIL、Python 解释器等 | **任意** | `python -m fastokens.observe sample` |
| `layers` | 把一次调用拆成：调用方 / transformers、绑定层、等 GIL、建 Python 列表、Rust 各阶段 | **观测版** | `python -m fastokens.observe layers` |

### 5.1 serve：仪表盘、/json、/metrics

在要观察的进程里：

```python
import fastokens.observe as obs
server = obs.serve({"glm": tok}, port=9464)    # 后台线程；port=0 自动选端口
```

- `http://127.0.0.1:9464/`：仪表盘（每秒刷新：吞吐、各阶段耗时、各路径、缓存命中、线程、最慢的请求）。右上角选统计窗口 1 / 10 / 60 秒，默认 10 秒，窗口越长数字越稳；普通版的仪表盘没有数据可显示；
- `/json`：完整快照；
- `/metrics`：Prometheus 文本格式，可以直接接 Prometheus / Grafana；
- `POST /reset-slow` 清空"最慢请求"列表，`POST /analyze` 供仪表盘使用。

注意：

- `serve()` 的第一个参数是要列出的分词器（`{名字: 分词器}` 或单个分词器），只在调用时确定一次。不传就只有进程级的数据（阶段、路径、线程、接口），没有缓存。
- 默认只监听 127.0.0.1，没有鉴权，不要暴露到公网；要从别的机器看，用 SSH 端口转发。
- `server.shutdown()` 停止。

### 5.2 top

```bash
python -m fastokens.observe top --url http://127.0.0.1:9464
```

需要观测版（普通版只打印 "nothing to show"）。每隔 `--interval` 秒（默认 2）打印一个窗口，包括：每秒编码数和 MB/s、平均和 p99 延迟、各路径和各阶段的耗时占比、各缓存的命中率、线程池和线程、Python 接口的 Rust / 等 GIL / Python 拆分。`--once` 只打印一次后退出。

### 5.3 profile

对一个开了 `serve` 的进程采一个窗口（默认 10 秒），打印完整报告，包括窗口内最慢的几个请求和它们的阶段拆分：

```bash
python -m fastokens.observe profile --url http://127.0.0.1:9464 --seconds 10
```

`--slowest N` 列出最慢的 N 个，`--keep-slow` 不先清空最慢列表，`--json` 输出分析数据而不是文字。也可以在自己的代码里包住一段：

```python
with obs.profile({"glm": tok}) as p:
    ...                       # 要分析的工作
print(p.report())             # p.data 是同样内容的字典
```

### 5.4 dump、analyze、report

```bash
python -m fastokens.observe dump --url http://127.0.0.1:9464 > snap.json
```

计数器只增不减，分析都基于两个快照之差：

```python
a = obs.snapshot({"glm": tok}); ...; b = obs.snapshot({"glm": tok})
print(obs.report(a, b))       # 文字报告；obs.analyze(a, b) 是字典
```

### 5.5 sample：CPU 采样（任意变体）

```bash
python -m fastokens.observe sample --pid <进程号> --seconds 10 --out prof
```

- 也可以启动一个命令并在热身 `--delay` 秒后采样：`python -m fastokens.observe sample --seconds 10 -- python my_bench.py`；或重新分析以前的原始输出：`--from prof.raw.txt`。
- `--out prof` 写出 `prof.txt`（报告）、`prof.json`、`prof.folded`（折叠栈，可直接给 flamegraph.pl / speedscope 画火焰图）和 `prof.raw.txt`（采样器原始输出）。Linux 上还会多留下一个 `prof.raw.txt.data`：`perf record` 的原始数据，`--from` 只读 `prof.raw.txt`，用不到它，可以删掉。macOS 上没有这个文件。
- 报告怎么读：
  - "where the CPU goes"：在 CPU 上的样本按栈顶函数分类，如 `stage: BPE`、`regex engine`、`allocator`、`binding: building Python lists / objects`、`binding: input str -> UTF-8`、`Python interpreter`、`GIL wait`；
  - "called from"：分配器、memcpy、正则引擎的时间分别是哪个阶段调用的；
  - "hottest functions"：自身耗时最多的函数；
  - "busiest threads"：最忙的线程。
- **macOS**：用系统自带的 `/usr/bin/sample`，不需要安装和权限。
- **Linux**：用 `perf record -g` 加 `perf script`。前提：
  1. 装 `perf`，Ubuntu 上是 `sudo apt-get install linux-tools-common linux-tools-$(uname -r)`，其他发行版装与内核版本对应的 `perf` 包；
  2. 有权限：`sudo sysctl -w kernel.perf_event_paranoid=1`，或用 root 运行；
  3. **容器里**：`perf_event_paranoid` 是内核（宿主机）的设置，要在宿主机上改，或在特权容器里改（会改到宿主机 / 虚拟机）；容器要有 `CAP_PERFMON`（内核 5.8+）或 `CAP_SYS_ADMIN`（如 `docker run --cap-add PERFMON` 或 `--privileged`），Docker 默认的 seccomp 规则在没有这两个权限时会拦下 `perf_event_open`；容器里的 `perf` 版本最好和内核一致。不满足时命令会报 "no CPU sampler" 或 perf 的权限错误，不影响进程本身。实测：Mac 上的 Docker Desktop（linuxkit 内核 6.12），`python:3.12-slim` 容器加 `--privileged`，`apt-get install linux-perf`（perf 6.12）、在容器里 `sysctl -w kernel.perf_event_paranoid=1`（原值 2）之后可以采样。
- **Linux 已验证**：`--help` 和模块说明里还写着 "Linux perf, unverified"，这是旧的说明。`1a0f7e2` 的 A3 冒烟测试已在 GitHub 的 ubuntu-24.04-arm 机器上跑通（内核 6.17、perf 6.17.13；3 秒约 3000 个样本，几乎全部在 CPU 上，能分到各阶段）。
- **Linux 已知问题**：PCRE2 即时编译出的正则代码在 perf 里显示为 `[unknown]`（`perf-<pid>.map`），被归进 `other`，在 A3 冒烟测试的编码循环里占 35–40%，其中大部分实际属于正则引擎（macOS 上这些帧会算给调用方，归入正则引擎）。在 Linux 上读报告时，把 `other` 的大部分当作正则匹配。
- 采样不需要观测版，也能分析 HF `tokenizers` 的进程，方便对照。

### 5.6 layers：一次调用的时间在哪一层（观测版）

```bash
python -m fastokens.observe layers --tokenizer tokenizer.json --inputs prompts.json --calls 200
```

- `--inputs`：字符串组成的 JSON 列表（嵌套列表会被展开），或一个文本文件（整个文件当一条输入）。
- `--via transformers --tokenizer <模型目录或 Hub id>`：先 `patch_transformers()`，再测 `tok(text)["input_ids"]`。默认 `--via native` 测 `Tokenizer.encode(text).ids`。
- 输出各层每次调用的毫秒数和占比：调用方和 transformers（含输入 `str` 转 UTF-8、释放上一次的结果）、绑定层的 Python 部分、等 GIL、建 Python 列表、Rust 各阶段。Python 里是 `obs.layers(fn, calls=...)` 和 `obs.layers_report(...)`。
- 普通版上会报错 "layers() needs fastokens built with the `observe` feature"。

## 6. 在 vLLM 里用分析功能

### 6.1 sample：直接按进程号采

`sample` 从外面采样，不需要改 vLLM，任意变体都能用。先找到 API 进程的进程号：vLLM 的日志里每个子进程的行带有 `(ApiServer_0 pid=12345)`；或者：

```bash
pgrep -f "vllm serve" -l
```

然后在有流量时采：

```bash
python -m fastokens.observe sample --pid 12345 --seconds 10 --out api0
```

报告里能看到 fastokens 各阶段、Jinja 模板渲染（归在 Python 解释器里）、GIL 等待的占比。

### 6.2 仪表盘、top、profile：用一个小插件

仪表盘、`top`、`profile` 都要求进程里调用了 `observe.serve()`，main 上没有环境变量能让它自动启动；而且 vLLM 的分词器（包括拷贝池里的每份拷贝）是在启动过程中才创建的。仓库里的 [`examples/vllm_observe/`](../examples/vllm_observe/) 是一个 vLLM general plugin，解决这两件事：

- 每个 vLLM 进程一创建 fastokens 分词器，它就在这个进程里启动 `serve()`，列出这个进程的所有分词器，每份各自的缓存。名字说明它是怎么来的：`tok0-loaded` 是加载的分词器，`tok1-copy-of-tok0` 是从 `tok0` 深拷贝出来的一份；
- 端口从 `FASTOKENS_OBSERVE_PORT` 开始找第一个空闲的，日志里会打印每个进程拿到的端口；`/json` 里也有 `pid`；
- 不设 `FASTOKENS_OBSERVE_PORT` 时什么都不做。

装进 vLLM 所在的环境：

```bash
python -m pip install ./examples/vllm_observe
```

启动 vLLM 时加上端口：

```bash
VLLM_USE_FASTOKENS=1 FASTOKENS_OBSERVE_PORT=9464 FASTOKENS_SEGMENT_CACHE=256 FASTOKENS_INPUT_CACHE=64 vllm serve <模型> --api-server-count 2
```

日志里每个创建了分词器的进程会打印一行：

```
(ApiServer_0 pid=…) INFO … fastokens observe: pid=… serving http://127.0.0.1:9465 (observe build: True)
```

然后：

```bash
python -m fastokens.observe top --url http://127.0.0.1:9465
```

说明：

- 实测（vLLM 0.23，`--api-server-count 2`）：两个 API 进程拿到 9464、9465，引擎进程（EngineCore）也创建了一个分词器、拿到 9466（它不处理聊天请求，缓存一直是空的）；`vllm serve` 的主进程没有创建。谁拿到哪个端口取决于启动先后，以日志为准。
- 每个 API 进程里列出 4 个分词器（默认 `--renderer-num-workers 1`）：`tok1-copy-of-tok0`、`tok2-copy-of-tok0` 是渲染线程池的两份拷贝，处理请求，各有自己的缓存（具体用到哪几份不固定：有的运行里两份交替使用，有的运行里分词全落在其中一份，另一份在仪表盘里显示为空闲）；`tok0-loaded` 是 vLLM 加载的原始分词器（请求都交给拷贝），`tok3-loaded` 是 vLLM 为读取聊天模板另外加载的一份，这两个不处理请求，缓存一直是空的，不占缓存内存，仪表盘把它们收在"空闲的实例"里。vLLM 走的是 `encode_batch` 接口，大多记在 `segments_cached` 路径（前缀缓存复用后只剩一小段的记在 `fused_split`，完全相同的请求记在 `prefix_exact`）。
- 两个缓存都开时，先看前缀缓存的复用比例（`top` 里 `reused … of input bytes`）：前缀缓存先把历史复用掉，段缓存只处理剩下的部分，所以它的命中率会显得很低，这是正常的。
- 插件不改变分词结果；它在每个新分词器创建时重启一次本进程的观测服务（只在启动时发生几次）。
- `FASTOKENS_OBSERVE_HOST` 可以改监听地址（默认 127.0.0.1，没有鉴权）。
- vLLM 默认加载所有已安装的 general plugin；如果用 `VLLM_PLUGINS` 限定了插件列表，要把 `fastokens_observe` 加进去。

## 7. 常见问题和排查

| 现象 | 原因和处理 |
|---|---|
| 设了 `FASTOKENS_SEGMENT_CACHE`，但 `segment_cache` 是 `null` | 变量在分词器创建之后才设；或进程没有继承到（检查 vLLM 子进程的环境）；或数值是 0 / 不是整数 |
| vLLM 里好像没用上 fastokens | 没设 `VLLM_USE_FASTOKENS=1`；或环境里装的是别的 fastokens：用 1.5 节的命令看路径和变体 |
| `observe.enabled()` 是 `False` | 装的是普通版；要命中率就换观测版（1.1） |
| `layers` 报错要 observe 构建；`top` / `profile` 只打印 "nothing to show" | 同上 |
| 普通版看不到命中率 | 设计如此，只有条数和字节；用观测版 |
| 内存比预算大很多 | 拷贝份数 × API 进程数（2.4、3.2）；段缓存最多 1.25 倍（2.7）；前缀缓存默认 2048 MiB 是每份拷贝的预算 |
| 命中率低 | 段缓存：预算小于工作集的 2.3 倍或小于最大段的 64 倍（2.3）；前缀缓存：条数少于活跃会话数（3.2）；没有会话亲和（2.4） |
| `sample` 在 Linux 上报 "no CPU sampler" 或权限错误 | 装 `perf`、设 `perf_event_paranoid`，容器里加权限（5.5） |
| Linux 报告里 `other` 很高 | PCRE2 即时编译代码的已知归类问题（5.5） |
| 插件端口没打开 | 没设 `FASTOKENS_OBSERVE_PORT`；进程没有创建 fastokens 分词器（例如引擎进程）；64 个端口都被占用（日志有警告） |
| 想回到不开缓存 / 不用 fastokens | 去掉对应变量重启；或去掉 `VLLM_USE_FASTOKENS`、卸载本 fork 的 wheel（2.8） |

## 8. 环境变量一览

| 变量 | 作用 | 默认 |
|---|---|---|
| `FASTOKENS_SEGMENT_CACHE=<MiB>` | 段缓存预算，开启段缓存 | 关闭 |
| `FASTOKENS_INPUT_CACHE=<条数>` | 前缀缓存条数，开启前缀缓存 | 关闭 |
| `FASTOKENS_INPUT_CACHE_MIB=<MiB>` | 前缀缓存（含特殊标记的部分）的字节预算 | 2048 |
| `FASTOKENS_BPE_THREADS=<数量>` | 单个请求内部并行的线程数 | 全部逻辑核（Apple Silicon 上为性能核数） |
| `VLLM_USE_FASTOKENS=1` | vLLM 0.23 起自带：让 vLLM 使用 fastokens | 关闭 |
| `FASTOKENS_OBSERVE_PORT=<端口>` | 第 6.2 节的插件：从这个端口起为每个进程开观测服务 | 不开 |
| `FASTOKENS_OBSERVE_HOST=<地址>` | 同上，监听地址 | 127.0.0.1 |

都在创建分词器（或进程启动）时读取。

## 9. 核对和实测记录

### 9.1 对着代码核对的

- 环境变量的解析：`src/segment_cache.rs` 的 `from_env`，`src/lib.rs` 的 `input_cache_capacity_from_env`、`prefix_cache_max_bytes_from_env`。`0`、不是数字都表示关闭；`FASTOKENS_INPUT_CACHE_MIB` 不设时是 2048。
- Python 的 `Tokenizer` 上没有 `enable_segment_cache`、`enable_input_cache`、`segment_cache_stats`。
- 命令行只有 `top`、`profile`、`dump`、`sample`、`layers` 五个子命令；`serve`、`analyze`、`report`、`profile()`、`layers()` 是 Python 接口（`python/fastokens/observe.py`）。
- vLLM 0.23 的源码：
  - `VLLM_USE_FASTOKENS`：`vllm/envs.py`、`vllm/tokenizers/registry.py`、`vllm/tokenizers/fastokens.py`；
  - 拷贝池份数 `--renderer-num-workers` + 1，借空时再拷一份：`vllm/renderers/hf.py`、`vllm/tokenizers/hf.py`。

### 9.2 本机实测（M2 Pro，macOS）

用的是普通版和观测版各一个 wheel：从 `49a99e8` 编出，Rust 和 Python 代码与 `1a0f7e2` 完全相同（`1a0f7e2` 只改了 CI 和文档）。每个都实际跑过：

- 1.5 节确认变体的命令；
- 2.2 / 3.1 节的环境变量：`256`、`0`、`abc`，前缀缓存的条数和字节预算，默认 2048 MiB；
- 2.5、2.6、3.4 节的 Python 片段：经 transformers（`patch_transformers()` 加 GLM-5.2）；
- 5.1 节的 `/`、`/json`、`/metrics`、`POST /reset-slow`；
- 5.2–5.4 节的 `top --once`、`profile`、`dump`，以及 Python 的 `with obs.profile(...)` 和 `obs.report()`；
- 5.5 节的 `sample --pid`、`sample -- 命令`、`sample --from` 及其写出的四个文件（macOS；Linux 上多一个 `.raw.txt.data`，见 5.5 节）；
- 5.6 节的 `layers`：`--via native` 和 `--via transformers`；
- 第 0 节第 6 步的一行命令。

普通版上，`top` / `profile` / `report()` 打印 "nothing to show"，`layers` 报错，那行命令报 `KeyError`，文中都已写明。

### 9.3 vLLM 实测

- 环境：vLLM 0.23.0 CPU 版，模型是 Qwen3-0.6B 的结构（1 层、dummy 权重），GLM-5.3 分词器，`--api-server-count 2`。
- 开关：`VLLM_USE_FASTOKENS=1`、`FASTOKENS_SEGMENT_CACHE=256`、`FASTOKENS_INPUT_CACHE=16`，第 6.2 节的插件加 `FASTOKENS_OBSERVE_PORT=9464`。
- 流量：96 个多轮渲染请求（`/v1/chat/completions/render`）和 1 个经过引擎的 `/v1/chat/completions`，全部返回 200。
- 插件：两个 API 进程和引擎进程各自打开了一个端口，`/`、`/json`、`/metrics` 都能访问；每个 API 进程有 4 个分词器，两份拷贝的段缓存和前缀缓存都有写入和命中。
- 工具：在持续流量下，`top --once`、`profile --seconds 4`（窗口内 348 次编码/秒、`segments_cached` 路径、`encode_batch` 接口）、`dump`、`sample --pid <API 进程>` 都正常输出。
- 这次的流量是合成的，目的只是验证功能，其中的命中率不代表真实负载。

### 9.4 Linux

- GitHub `ubuntu-24.04-arm`（`1a0f7e2` 的 A3 构建，运行 39193271458）：两个变体的 wheel 都通过了冒烟测试，内容是两种缓存的四种开关组合在两种分词器上逐 token 一致、观测版的 `layers` 和快照、`observe sample` 走 perf 6.17.13 采样。
- 本机 Docker Desktop（linuxkit 6.12 内核，`python:3.12-slim`，aarch64）：用同一次运行的两个 A3 wheel，`pip install`、确认变体、冒烟测试都通过；`--privileged` 容器里装 Debian 的 `linux-perf` 6.12 后 `sample` 能采样，3011 个样本，`other` 占 32%（PCRE2 即时编译代码，见 5.5）。

### 9.5 没能实测的

- 真实的 A3 机器（第 0 节的步骤只在 GitHub 的 arm64 机器和本机容器里跑过；`gh` 命令在本机跑过）。
- GPU 上的 vLLM，以及 vLLM 0.30 上的插件。
- 插件是用 `pip install --target` 装进单独目录、放进 `PYTHONPATH` 测的，没有直接装进 vLLM 的环境；效果相同，但安装命令本身没在 vLLM 环境里执行过。
- 文中的性能数字都引用已有的报告，本次没有重新测性能。
