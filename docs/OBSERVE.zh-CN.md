# fastokens 内部剖析（observe）使用说明

`observe` 是一个 cargo 特性，默认关闭。打开后，fastokens 会在内部记录下面这些信息，供你自己剖析：

- 编码流水线**每个阶段**花了多少时间，按编码路径分开统计；
- **每一层缓存**的效果：BPE 词缓存、segment cache、前缀缓存；
- 线程池和每个线程在做什么；
- **最慢的请求**，以及它们各阶段的耗时；
- Python API 调用的时间分别花在 Rust、等 GIL 和 Python 哪一部分。

关闭时，所有埋点都是空函数，或者是零大小、不带析构的类型，编译出来的代码和原来一样。

开启时的开销实测（M2 Pro，GLM-5.3 agent 流量，1920 个请求，三个构建交替跑，取每个构建 3 轮中的最好值）：和未加观测的版本相比在 −2.0% 到 +1.3% 之间，属于噪声。

## 1. 能剖析什么

### 1.1 编码各阶段（每次编码都计，按路径汇总）

| 阶段 | 含义 |
|---|---|
| `added_tokens` | 在输入里匹配特殊 token（`<\|im_start\|>` 等），并按它们切开 |
| `normalize` | 归一化（NFC 等；没有归一化器的模型为 0） |
| `prefix_cache` | 前缀缓存：查找共享前缀的旧输入，以及存入新输入 |
| `pretokenize` | 单独的预分词（正则 `Split`、`ByteLevel`）。扫描器路径上它和 BPE 融合在一起，计入 `model` |
| `model` | BPE 模型（扫描器路径上包含融合的扫描） |
| `cache_lookup` | segment cache 查找（含哈希） |
| `cache_assemble` | 把未命中的分段写入缓存，并把命中部分和新结果拼起来 |
| `post_process` | 后处理（加 BOS/EOS 等特殊 token） |
| `other` | 总时间减去以上各阶段之和（内存分配、拼接等） |

各阶段互不重叠。segment cache 未命中时，那部分分词的时间仍计入 `pretokenize` / `model`。

**编码路径**：
- `scan_whole`：整段用扫描器；
- `scan_splits`：按特殊 token 分段扫描，聊天模板走这条；
- `segments_cached`：经过 segment cache；
- `fused_split`：`Split` 预分词与模型融合，非扫描器模型走这条；
- `general`：通用流水线。

每条路径另外还记录：输入字节数、输出 token 数、延迟分布和大小分布。

### 1.2 缓存

| 缓存 | 记录了什么 |
|---|---|
| **BPE 融合预分词缓存**（每线程一份；扫描器和 `fused_split` 路径用） | 查找次数、未命中次数（每次未命中要做一次 BPE merge）、merge 的字节数和耗时、无法缓存的长片段数（超过 15 字节）、缓存满了被清空的次数。由此算出命中率、每次 merge 的耗时、merge 占模型阶段的比例 |
| **BPE 通用路径缓存**（`general` 路径用） | 调用次数、整段本身就是一个 token 的次数、线程缓存命中、跨线程共享缓存命中、merge 次数和耗时 |
| **segment cache**（每个分词器一份，`FASTOKENS_SEGMENT_CACHE`） | 命中/未命中（次数和字节）、未命中分词所花的时间、**估算节省的时间**（命中字节数 × 本窗口未命中时每字节的分词成本；未命中多为冷数据，所以这个估算偏高）、16 个分片的新/旧代占用、代际轮换、淘汰、旧代提升、插入、覆盖、哈希碰撞、被拒的超大分段、分片锁竞争和等待时间、分段大小分布（每个线程每 16 次查找抽样 1 次） |
| **前缀缓存**（每个分词器一份，`FASTOKENS_INPUT_CACHE=<容量>`） | 查找次数，其中完整命中、前缀复用、未命中各多少；复用了多少字节和 token，占输入字节的比例。这些总数是纯文本前缀缓存（只在扫描器整段路径上生效）和推理请求前缀缓存（见下一行）之和 |
| **前缀缓存 · 推理请求**（`prefix_cache.chat`；含特殊标记的输入，即 chat template 输出） | 查找次数和完整命中/前缀复用/未命中；查找、复用、重新编码的字节数（复用 + 重新编码 = 查找），复用的 token 数；切点回退（找到了共享前缀，但较晚的切点被规则排除，退回更早的切点）和无可用切点；绕过次数（短于 4 KiB 的不查找；不含特殊标记的查找后才发现，交给纯文本缓存，不计入这里的查找：扫描器路径上纯文本缓存会再查一次并计数，其他路径（HF 正则类分词器）上这次查找哪边都不计）；占位 id 冲突回退（预期为 0）；写入、同一会话替换旧条目、按最近最少使用淘汰的条目数和字节数；单条超过预算 1/64 未缓存的次数和字节数；改词表或改 normalizer 时清空的条目数；条目数、缓存自报字节数、字节预算（`max_bytes`）、索引块数；锁竞争次数和等待时间。不变式：条目数 = 写入 − 替换 − 淘汰 − 清空。查找、选切点、写入的耗时算在 `prefix_cache` 阶段，切分特殊标记算在 `added_tokens` 阶段 |

### 1.3 线程、线程池、Python

- **并行 job**：按来源分类（长文本扫描分块、聊天模板分段、通用流水线分块、正则切分、encode_batch），记录 job 数、任务数、墙钟时间和任务时间之和。两者之比就是**平均并行度**。
- **每个线程**：调用线程（Python 线程会带上自己的名字，如 `py:load-3`）、BPE 线程池（`fastokens-N`）、rayon 全局池（`rayon-N`）各自的忙碌占比、编码次数、字节数和 token 数、各路径次数、各阶段耗时、任务数、在 Python API 里的时间、BPE 缓存计数，以及 OS 线程号。
- **Python API**：每个接口记录调用次数、延迟分布，以及时间的三段拆分：Rust（含状态锁）/ 等回 GIL / Python。`encoding_ids` 和 `encoding_lists` 是读取 `Encoding.ids`、`attention_mask` 等属性时构建 Python 列表的开销。另外还记录状态锁的读/写阻塞。
- **最慢的编码**：保留最慢的 32 次，每次都带耗时、字节数、token 数、路径、线程和各阶段耗时；可以清空，重新开始统计。

## 2. 构建

```bash
maturin build --release --features observe          # Python wheel
cargo build --release --features observe            # Rust
```

不加 `--features observe` 就是普通版：`fastokens.observe.enabled()` 返回 `False`，所有接口照常可用，只是没有数据。

## 3. 使用

### 3.1 剖析一段代码

```python
import fastokens.observe as obs

with obs.profile({"glm": tok}) as p:     # 进入时清空"最慢编码"列表
    ...                                  # 你的负载
print(p.report())                        # 文字报告
p.data                                   # 同样的内容，dict 形式（analyze() 的结果）
```

`tok` 可以是 `fastokens._native.Tokenizer`、`patch_transformers()` 之后的 transformers 分词器，或者它的 shim。

### 3.2 观察一个正在运行的进程

在做分词的进程里：

```python
obs.serve({"glm": tok}, port=9464)   # 后台线程，只监听 127.0.0.1；port=0 自动选空闲端口
```

- `http://127.0.0.1:9464`：实时仪表盘，每秒刷新，不依赖任何外部资源。
- `GET /json`：完整快照。
- `GET /metrics`：Prometheus 文本格式。推理请求前缀缓存的指标以 `fastokens_prefix_cache_chat_` 开头（计数器带 `_total`，条目数、字节数、字节预算、索引块数是 gauge）。
- `POST /analyze`：请求体是上一次的快照，返回当前快照及两者之间的分析。
- `POST /reset-slow`：清空"最慢编码"列表。

另一个终端：

```bash
python -m fastokens.observe profile --url http://127.0.0.1:9464 --seconds 10           # 剖析 10 秒，打印报告
python -m fastokens.observe profile --url http://127.0.0.1:9464 --seconds 10 --json    # 同上，输出 JSON
python -m fastokens.observe top     --url http://127.0.0.1:9464                        # 每 2 秒刷新的文字界面
python -m fastokens.observe dump    --url http://127.0.0.1:9464                        # 原始快照
```

### 3.3 自带的压测驱动

```bash
cd examples/agent
python observe_load.py --threads 4 --cache-mib 256 --report-every 5
```

## 4. 怎么读

一个实际例子：GLM-5.3，main，4 个线程跑 agent 流量。

```
where encode time goes (ms per encode; % of the encode):
path       mean   added_tokens   pretokenize     model         cache_lookup  cache_assemb  other
fused_split 3.648  0.174  4.8%   2.911 79.8%    0.548 15.0%                                0.014     ← 不开缓存
segments_cached 0.236 0.152 64.5%                             0.050 21.3%   0.020  8.4%   0.014     ← 开缓存
API     mean 1.48 ms = rust 0.72 + GIL wait 0.76 + python 0.00;  encoding_ids 0.35 ms/次
BPE pretoken cache: hit rate 99.2%, 0.67 us per merge, merge 占模型时间 69.5%
```

从这个例子能直接读出：
- 不开缓存时，80% 的时间花在**正则预分词**上，BPE 本身只占 15%；
- 开缓存后，每次编码从 3.6 ms 降到 0.24 ms，这时剩下的大头是**特殊 token 匹配**（64.5%）；
- Python 线程每次调用有一半时间在**等 GIL**，读 `.ids` 构建列表每次还要再花 0.35 ms。

| 现象 | 说明 | 方向 |
|---|---|---|
| API 平均延迟远大于编码路径的平均延迟 | 时间花在 Rust 之外 | 看"等 GIL"和 Python 两列，以及 `encoding_ids` |
| `pretokenize` 占比最高 | 正则切分是瓶颈（非扫描器模型） | 换走扫描器路径的模型或版本 |
| `added_tokens` 占比最高 | 特殊 token 匹配是瓶颈（通常是开缓存之后） | 优化特殊 token 的匹配 |
| BPE 命中率低、merge/s 高 | 词缓存总是冷的：线程太多太新，或者缓存被清空 | 看"缓存清空"次数，以及线程是否反复新建 |
| 不可缓存片段很多 | 超过 15 字节的长片段（长数字串、长标识符等）每次都要 merge | 了解即可 |
| segment cache 命中率低，轮换和淘汰多 | 缓存太小 | 加大 `FASTOKENS_SEGMENT_CACHE` |
| 超大被拒 > 0 | 有分段的成本超过单段上限（约为预算的 1/64） | 预算至少给到"最大分段成本 × 64" |
| 缓存锁竞争 > 2% | 很多线程同时打同一个缓存 | 看是哪些线程在打（线程表） |
| 并行度接近 1 | 并行 job 没能真正同时运行（核不够，或任务太小） | 看线程池线程的忙碌占比 |
| 状态锁写等待 > 0 | 有线程在其他线程编码时修改了配置 | 尽量在启动时配置好 |
| 最慢编码里某个阶段特别大 | 长尾请求就卡在这个阶段 | 对照它的字节数和路径 |

## 5. 局限

- 快照不是一个原子时刻，但每个计数器都单调递增，算出来的速率和比例是准的。
- 阶段时间是请求关键路径上的墙钟时间。并行 job 里各任务的时间，另外记在"任务时间"和"并行度"里。
- rayon 线程池内部的自旋 / 休眠看不到；"任务时间"包含等待嵌套并行任务的时间。
- 只有一个输入的 `encode_batch` 会被 rayon 直接在调用线程上执行，这时算作调用线程的编码。
- 线程表最多记 255 个线程，之后的合并为"(other threads)"。
- p50/p99 来自 log2 直方图，显示的是所在桶的上界。
- 仪表盘只监听 127.0.0.1，没有鉴权；不要暴露到公网。
