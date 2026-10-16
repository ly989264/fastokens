# fastokens 观测（observe）使用说明

观测代码由 cargo 特性 `observe` 控制，默认关闭。关闭时所有埋点都是空函数或零大小类型，编译结果和原来一样。在 M2 Pro 上用 GLM-5.3 的 agent 流量（1920 个请求，三种构建交替各跑 3–4 轮）实测：
- 关闭时与未加观测的版本相差 −0.6% 到 +2.9%，属于噪声；
- 开启时最坏的场景是"开 segment cache 的 encode_batch，10 个线程同时查缓存"，慢 0.9%–1.4%；其余场景看不出开销。

## 1. 能看到什么

| 层面 | 内容 | 回答的问题 |
|---|---|---|
| Python API | 每个 API（encode / encode_ordinary / encode_segments / encode_batch / encode_batch_flat / decode / decode_batch）的每秒调用、MB/s、平均和 p50/p99 延迟、是否释放了 GIL，以及 **Rust 干完活后重新拿到 GIL 前等了多久** | 慢在 Rust 里，还是慢在 Python/GIL 上？ |
| 编码路径 | 每次编码走了哪条路：`scan_whole`（整段扫描）、`scan_splits`（按特殊 token 分段扫描）、`segments_cached`（经过 segment cache）、`fused_split`（Split 预分词与模型融合，非扫描器模型）、`general`（通用流水线） | Rust 部分真正花了多少时间；缓存有没有生效 |
| 并行 job | 按来源统计 job 数、任务数、耗时：`scan`（长文本扫描分块）、`chat_splits`（聊天模板分段）、`pretokenized`（通用流水线分块）、`regex_split`（超长输入的正则切分）、`batch`（encode_batch 每条） | fastokens 什么时候用了多线程 |
| 每个线程 | 调用线程（Python 线程自动带名字，如 `py:load-3`）、BPE 线程池（`fastokens-N`）、rayon 全局池（`rayon-N`，encode_batch 用它）各自的忙碌占比、编码次数和字节数、各路径次数、任务数、在 Python API 里的时间、OS 线程号 | 谁在干活、负载均不均衡 |
| 状态锁 | Python 绑定层 `RwLock` 读/写阻塞的次数和时间 | 有没有线程在运行中改配置 |
| Segment cache（每个分词器） | 占用 / 约预算、条目、分段和字节命中率、16 个分片各自的新代/旧代占用、代际轮换、淘汰的条目和字节、旧代提升、插入、覆盖、哈希碰撞、**每个分片的锁竞争次数和等待时间**、命中/未命中分段大小分布（每个线程每 16 次查找抽样 1 次）、因超过单段上限被拒绝的分段、未命中分词时间 | 缓存够不够大、多线程下锁是否成为瓶颈 |
| 进程 | CPU（核）、RSS | 总体资源 |

计数器只增不减，页面和 `top` 上显示的都是"两次快照之差 ÷ 时间差"。

## 2. 构建

```bash
maturin build --release --features observe          # Python wheel
cargo build --release --features observe            # Rust
```

不加 `--features observe` 就是普通版：`fastokens.observe.enabled()` 返回 `False`，页面提示没有开启。

## 3. 使用

在做分词的那个进程里：

```python
import fastokens.observe as obs

tok = ...   # fastokens._native.Tokenizer，或 patch_transformers() 之后的 transformers 分词器
obs.serve({"glm": tok}, port=9464)   # 后台线程，只监听 127.0.0.1；port=0 自动选空闲端口
```

- 浏览器打开 `http://127.0.0.1:9464`：实时仪表盘，每秒刷新，不依赖任何外部资源。
- `curl http://127.0.0.1:9464/json`：完整快照。
- `curl http://127.0.0.1:9464/metrics`：Prometheus 文本格式，可以直接让 Prometheus 抓。
- 另一个终端：

```bash
python -m fastokens.observe top  --url http://127.0.0.1:9464          # 文字版，2 秒刷新
python -m fastokens.observe top  --url http://127.0.0.1:9464 --once   # 打印一个窗口
python -m fastokens.observe dump --url http://127.0.0.1:9464          # 原始 JSON
```

不起 HTTP 也可以直接在代码里取：

```python
a = obs.snapshot({"glm": tok}); ...; b = obs.snapshot({"glm": tok})
print(obs.report(a, b))
```

segment cache 在创建分词器时由 `FASTOKENS_SEGMENT_CACHE=<MiB>` 开启。线程池、线程、API 统计是整个进程共享的，缓存是每个分词器各一份。

自带的压测驱动（会同时起仪表盘）：

```bash
cd examples/agent
python observe_load.py --threads 4 --cache-mib 256
python observe_load.py --threads 4 --cache-mib 0 --api batch --report-every 5
```

## 4. 怎么读

| 现象 | 说明 | 怎么办 |
|---|---|---|
| API 平均延迟远大于编码路径平均延迟 | 时间花在 Rust 之外：等 GIL、建 Python 结果对象 | 看"等回 GIL"；减少同进程里其他占 GIL 的 Python 工作，或者开多个进程 |
| 编码路径是 `fused_split` 而不是 `scan_*` | 这个模型的预分词没有走扫描器快路径 | 正常，取决于模型的正则 |
| `regex_split` / `pretokenized` job 很多 | 超长输入在线程池上并行切分、分词 | 正常；单请求会用到多个核 |
| 缓存字节命中率低，代际轮换/s、淘汰 MiB/s 高 | 缓存太小，内容还没被复用就被挤掉了 | 加大 `FASTOKENS_SEGMENT_CACHE` |
| 超大被拒/s > 0 | 有分段的成本超过了单段上限（约预算的 1/64） | 预算至少给到"最大分段成本 × 64" |
| 缓存锁竞争 > 2% | 很多线程同时打同一个缓存（比如 encode_batch 用满 rayon 全局池） | 了解即可；锁等待时间也会一并显示 |
| 状态锁写等待 > 0 | 有线程在其他线程编码时调用了 `enable_padding` / `enable_truncation` / `add_tokens` 等 | 尽量在启动时配置好，运行中不要改 |
| 哈希碰撞 > 0 | 两段不同文本的 64 位哈希相同 | 不会出错（命中前会比对原文），只是少一次命中 |

## 5. 局限

- 快照不是一个原子时刻，但每个计数器单调递增，算出来的速率是准的。
- 线程表最多 255 个线程，之后的合并为"(other threads)"；已退出的线程标"已退出"，保留历史计数。
- rayon 线程池内部的自旋 / 休眠看不到；"任务时间"包含等待嵌套并行任务的时间。
- 只有一个输入的 `encode_batch` 会被 rayon 直接在调用线程上执行，此时算作调用线程的编码。
- p50/p99 来自 log2 直方图，显示的是所在桶的上界。
- 仪表盘只监听 127.0.0.1，没有鉴权；不要暴露到公网。
