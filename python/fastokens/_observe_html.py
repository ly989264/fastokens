"""The page `fastokens.observe.serve()` serves at `/`. Once a second it posts
its previous snapshot to `analyze` and shows the window's analysis, computed
by the same `fastokens.observe.analyze` that `report()` prints. Self-contained
(no network)."""

DASHBOARD = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fastokens observe</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#e4e3de; --bad:#c0392b;
        --s0:#2f6fdb; --s1:#8e5bd1; --s2:#2f9e9e; --s3:#d9822b; --s4:#2f9e6b; --s5:#b5476b; --s6:#9a7d2b; --s7:#5b7a99; --s8:#bdbbb3; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#151514; --fg:#ecebe6; --muted:#9a998f; --card:#1e1e1c; --line:#33322e; --bad:#ef6f61;
          --s0:#6c9cf0; --s1:#b08ae6; --s2:#5cc2c2; --s3:#e9a25a; --s4:#5cc293; --s5:#e07a9c; --s6:#c9ac5a; --s7:#8fa9c4; --s8:#5a5953; } }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg);
       font:13px/1.45 -apple-system, "PingFang SC", "Segoe UI", sans-serif; }
h1 { font-size:18px; margin:0 0 4px; }
h2 { font-size:14px; margin:0 0 8px; display:flex; justify-content:space-between; align-items:center; gap:8px; }
.sub { color:var(--muted); margin-bottom:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(440px, 100%), 1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; overflow-x:auto; min-width:0; }
.wide { grid-column:1/-1; }
.kpis { display:flex; flex-wrap:wrap; gap:6px 18px; margin-bottom:8px; }
.kpi b { display:block; font-size:17px; font-variant-numeric:tabular-nums; }
.kpi span { color:var(--muted); font-size:12px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:3px 6px; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:500; }
canvas { width:100%; height:70px; display:block; margin:6px 0; }
.legend { color:var(--muted); font-size:12px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin:0 4px -1px 10px; }
.warn { color:var(--bad); }
.stack { display:flex; height:14px; border-radius:3px; overflow:hidden; background:var(--line); min-width:160px; }
.stack i { display:block; height:100%; }
.bars { display:flex; align-items:flex-end; gap:3px; height:80px; margin:6px 0; }
.bars div { flex:1; display:flex; flex-direction:column-reverse; height:100%; background:color-mix(in srgb, var(--line) 50%, transparent); border-radius:2px; overflow:hidden; }
.bars i { display:block; width:100%; }
.note { color:var(--muted); font-size:12px; margin-top:6px; }
button { font:inherit; font-size:12px; padding:2px 10px; border:1px solid var(--line); border-radius:5px; background:var(--card); color:var(--fg); cursor:pointer; }
</style>
</head>
<body>
<h1>fastokens observe</h1>
<div class="sub" id="sub">连接中…</div>
<div class="grid">
  <div class="card"><h2>概览</h2><div class="kpis" id="overview"></div>
    <canvas id="c_cores"></canvas>
    <div class="legend"><i style="background:var(--s0)"></i>调用线程编码（核） <i style="background:var(--s3)"></i>线程池任务（核） <i style="background:var(--s4)"></i>进程 CPU（核）</div></div>
  <div class="card"><h2>Python API</h2><table id="api"></table>
    <div class="note">每次调用拆成：Rust（含状态锁）、等回 GIL（Rust 干完后重新拿 GIL）、Python（参数转换、建结果对象）。p50/p99 为 log2 直方图所在桶的上界。encoding_ids / encoding_lists 是读取 Encoding.ids 等属性时构建 Python 列表的开销。</div></div>
  <div class="card wide"><h2>编码各阶段耗时</h2><div class="legend" id="stage_legend"></div><table id="stages"></table>
    <div class="note">每次编码的平均毫秒数与占比。model 在扫描器路径上包含融合的扫描；other = 总时间 − 各阶段之和（内存分配、拼接等）。cache_lookup / cache_assemble 只出现在开了 segment cache 的路径上。</div></div>
  <div class="card"><h2>BPE 内部缓存</h2><div class="kpis" id="bpe"></div>
    <div class="note">融合预分词缓存：每个线程一份，按预分词片段查；未命中要做 BPE merge。超过 15 字节的片段无法缓存，每次都 merge。缓存满了会整体清空。</div></div>
  <div class="card"><h2>并行 job</h2><div class="kpis" id="pool"></div><table id="jobs"></table>
    <div class="note">并行度 = 各任务时间之和 ÷ job 墙钟时间（≈ 同时在干活的线程数）。</div></div>
  <div class="card wide"><h2><span>最慢的编码（窗口内）</span><button id="reset">清空，重新统计</button></h2><table id="slow"></table>
    <div class="note">进程内保留最慢的 32 次编码；这里只列出本窗口内结束的那些。</div></div>
  <div class="card wide"><h2>线程</h2><table id="threads"></table>
    <div class="note">忙碌 = 窗口内在 fastokens 里干活的时间占比（调用线程按编码计，线程池线程按任务计）；API 内 = 调用线程在 fastokens Python API 里的时间占比。</div></div>
</div>
<div id="caches"></div>
<script>
const HIST = 120;
const STAGES = ["added_tokens", "normalize", "prefix_cache", "pretokenize", "model", "cache_lookup", "cache_assemble", "post_process", "other"];
const SN = {added_tokens: "特殊 token", normalize: "归一化", prefix_cache: "前缀缓存", pretokenize: "预分词", model: "模型(BPE)",
            cache_lookup: "缓存查找", cache_assemble: "缓存拼装", post_process: "后处理", other: "其他"};
let prev = null;
const hist = {};
const $ = id => document.getElementById(id);
const fmt = (v, d=1) => v == null || !isFinite(v) ? "–" : v.toLocaleString(undefined, {maximumFractionDigits:d, minimumFractionDigits:d});
const pctf = v => v == null || !isFinite(v) ? "–" : (100*v).toFixed(1) + "%";
const mib = b => fmt(b / 1048576, 1) + " MiB";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function kpis(el, items) { el.innerHTML = items.map(([k, v, cls]) => `<div class="kpi"><b class="${cls||""}">${esc(v)}</b><span>${esc(k)}</span></div>`).join(""); }
function table(el, head, rows, raw) { el.innerHTML = "<tr>" + head.map(h => `<th>${esc(h)}</th>`).join("") + "</tr>" +
  rows.map(r => "<tr>" + r.map((c, i) => `<td>${raw && raw.includes(i) ? c : esc(c)}</td>`).join("") + "</tr>").join(""); }
function css(v) { return getComputedStyle(document.documentElement).getPropertyValue(v).trim(); }
function spark(canvas, series, colors, maxv) {
  const dpr = devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  if (!w || !h) return;
  canvas.width = w * dpr; canvas.height = h * dpr; const g = canvas.getContext("2d"); g.scale(dpr, dpr);
  const m = maxv || Math.max(1e-9, ...series.flat());
  g.strokeStyle = css("--line"); g.beginPath(); g.moveTo(0, h - 0.5); g.lineTo(w, h - 0.5); g.stroke();
  series.forEach((s, k) => { g.strokeStyle = css(colors[k]); g.lineWidth = 1.5; g.beginPath();
    s.forEach((v, i) => { const x = w * i / (HIST - 1), y = h - 2 - (h - 4) * v / m; i ? g.lineTo(x, y) : g.moveTo(x, y); }); g.stroke(); });
  g.fillStyle = css("--muted"); g.font = "11px sans-serif"; g.textAlign = "right"; g.fillText("max " + fmt(m, 2), w - 2, 11);
}
function push(key, v) { (hist[key] = hist[key] || []).push(v); if (hist[key].length > HIST) hist[key].shift(); return hist[key]; }
function stack(parts) {  // parts: [[stage, ms]]
  const total = parts.reduce((a, [, v]) => a + v, 0) || 1;
  return `<div class="stack">` + parts.filter(([, v]) => v > 0).map(([s, v]) =>
    `<i title="${esc(SN[s] || s)} ${fmt(v, 3)} ms" style="width:${100 * v / total}%;background:var(--s${STAGES.indexOf(s)})"></i>`).join("") + `</div>`;
}

function render(A, snap) {
  const P = snap.process;
  if (!P.enabled) { $("sub").innerHTML = '<span class="warn">这个 fastokens 构建没有开启 observe 特性（maturin build --features observe），没有可显示的计数。</span>'; return; }
  $("sub").textContent = `pid ${A.pid} · 窗口 ${A.window_s.toFixed(1)} s · BPE 线程池 ${A.pool.bpe_threads} 线程 · rayon 全局池 ${A.pool.global_threads} 线程 · 运行 ${fmt(P.uptime_s, 0)} s`;
  let callerBusy = 0, workerBusy = 0;
  for (const t of A.threads) { if (t.kind === "worker") workerBusy += t.busy; else callerBusy += t.busy; }
  const E = A.encodes;
  kpis($("overview"), [["进程 CPU（核）", fmt(A.cpu_cores, 2)], ["RSS", A.rss_bytes ? mib(A.rss_bytes) : "–"], ["编码/s", fmt(E.per_s, 1)],
    ["输入 MB/s", fmt(E.mb_per_s, 1)], ["token/s", fmt(E.tokens_per_s, 0)], ["调用线程编码（核）", fmt(callerBusy, 2)], ["线程池任务（核）", fmt(workerBusy, 2)]]);
  spark($("c_cores"), [push("caller", callerBusy), push("worker", workerBusy), push("cpu", A.cpu_cores ?? 0)], ["--s0", "--s3", "--s4"]);

  table($("api"), ["API", "调用/s", "平均 ms", "p99 ≤ ms", "Rust ms", "等 GIL ms", "Python ms", "GIL 释放"],
    Object.entries(A.api).map(([k, o]) => [k, fmt(o.per_s, 1), fmt(o.mean_ms, 3), fmt(o.p99_ms, 3), fmt(o.rust_ms, 3), fmt(o.gil_wait_ms, 3), fmt(o.python_ms, 3), pctf(o.gil_released)]));

  $("stage_legend").innerHTML = STAGES.map((s, i) => `<i style="background:var(--s${i})"></i>${esc(SN[s])}`).join(" ");
  table($("stages"), ["路径", "编码/s", "平均 ms", "p99 ≤ ms", "KB/次", "字节/token", "构成", ...STAGES.map(s => SN[s] + " ms")],
    Object.entries(A.paths).map(([k, p]) => {
      const parts = STAGES.map(s => [s, s === "other" ? p.other_ms : p.stage_ms[s]]);
      return [k, fmt(p.per_s, 1), fmt(p.mean_ms, 3), fmt(p.p99_ms, 3), fmt(p.mean_bytes / 1e3, 1), fmt(p.bytes_per_token, 2), stack(parts),
        ...parts.map(([s, v]) => v > 0 ? `${fmt(v, 3)} (${pctf(v / p.mean_ms)})` : "·")];
    }), [6]);

  const F = A.bpe.fused, G = A.bpe.general, dt = A.window_s;
  const items = [];
  if (F.lookups) items.push(["融合缓存命中率", pctf(F.hit_rate)], ["查找/s", fmt(F.lookups / dt, 0)], ["merge/s", fmt(F.misses / dt, 0)],
    ["每次 merge µs", fmt(F.merge_us_per_miss, 2)], ["merge 占模型时间", pctf(F.merge_share_of_model)], ["不可缓存片段", F.uncacheable.toLocaleString()],
    ["缓存清空", F.clears, F.clears ? "warn" : ""]);
  if (G.calls) items.push(["通用路径调用", G.calls.toLocaleString()], ["整段即 token", G.exact.toLocaleString()], ["线程缓存命中", G.thread_hits.toLocaleString()],
    ["共享缓存命中", G.shared_hits.toLocaleString()], ["merge", G.merges.toLocaleString()], ["通用命中率", pctf(G.hit_rate)]);
  if (!items.length) items.push(["状态", "窗口内没有 BPE 工作"]);
  kpis($("bpe"), items);

  const lw = A.lock_waits;
  let jobs = 0; for (const k in A.jobs) jobs += A.jobs[k].per_s;
  kpis($("pool"), [["并行 job/s", fmt(jobs, 1)], ["状态锁读等待/s", fmt(lw.read.per_s, 1), lw.read.per_s ? "warn" : ""], ["状态锁写等待/s", fmt(lw.write.per_s, 1), lw.write.per_s ? "warn" : ""]]);
  const KN = {scan: "长文本扫描分块", chat_splits: "聊天模板分段", pretokenized: "通用流水线分块", regex_split: "正则切分", batch: "encode_batch 每条"};
  table($("jobs"), ["job 来源", "job/s", "每 job 任务", "平均 ms", "并行度"], Object.entries(A.jobs)
    .map(([k, v]) => [KN[k] || k, fmt(v.per_s, 1), fmt(v.tasks_per_job, 1), fmt(v.mean_ms, 3), fmt(v.parallelism, 2)]));

  table($("slow"), ["耗时 ms", "KB", "token", "路径", "线程", "构成", "主要阶段"], A.slowest.slice(0, 15).map(s => {
    const parts = STAGES.filter(x => x !== "other").map(x => [x, (s.stage_ns[x] || 0) / 1e6]);
    const other = Math.max(0, s.ns / 1e6 - parts.reduce((a, [, v]) => a + v, 0)); parts.push(["other", other]);
    const top = parts.slice().sort((a, b) => b[1] - a[1]).slice(0, 3).filter(([, v]) => v > 0).map(([x, v]) => `${SN[x]} ${fmt(v, 2)}`).join("，");
    return [fmt(s.ns / 1e6, 2), fmt(s.bytes / 1e3, 1), s.tokens.toLocaleString(), s.path, s.thread, stack(parts), top];
  }), [5]);

  table($("threads"), ["线程", "类型", "OS tid", "忙碌", "API 内", "编码/s", "MB/s", "主要路径", "任务/s"],
    A.threads.map(t => [t.name + (t.alive ? "" : " (已退出)"), t.kind === "worker" ? "worker" : "调用方", t.os_tid, pctf(t.busy),
      t.in_api == null ? "–" : pctf(t.in_api), fmt(t.encodes_per_s, 1), fmt(t.mb_per_s, 1), t.main_path || "–", fmt(t.tasks_per_s, 1)]));

  const box = $("caches");
  for (const [name, T] of Object.entries(A.tokenizers)) {
    const id = "tok_" + name.replace(/[^A-Za-z0-9_-]/g, "_");
    let card = document.getElementById(id);
    if (!card) { card = document.createElement("div"); card.className = "card"; card.id = id; card.style.marginTop = "12px";
      card.innerHTML = `<h2>缓存 · ${esc(name)}</h2><div class="legend">Segment cache</div><div class="kpis seg"></div><canvas></canvas>
        <div class="legend"><i style="background:var(--s0)"></i>分段命中率 <i style="background:var(--s3)"></i>字节命中率</div>
        <div class="legend" style="margin-top:8px">各分片占用（满格 = 该分片预算）</div><div class="bars"></div>
        <div class="legend" style="margin-top:8px">前缀缓存（FASTOKENS_INPUT_CACHE）</div><div class="kpis pre"></div>`; box.appendChild(card); }
    const S = T.segment_cache, seg = card.querySelector(".seg");
    if (!S) kpis(seg, [["状态", "未开启（FASTOKENS_SEGMENT_CACHE）"], ["扫描器快路径", T.scanner_path ? "是" : "否"]]);
    else {
      const it = [["占用", `${mib(S.bytes)} / ~${mib(S.approx_max_bytes)}`], ["条目", S.entries.toLocaleString()], ["单段上限", fmt(S.max_segment_cost / 1024, 1) + " KB 成本"]];
      if (S.hits != null) {
        it.push(["分段命中率", pctf(S.hit_rate)], ["字节命中率", pctf(S.byte_hit_rate)], ["估算节省 ms/s", fmt(S.est_saved_ms / dt, 1)],
          ["未命中分词 ms/s", fmt(S.miss_tokenize_ms / dt, 1)], ["代际轮换", S.rotations], ["淘汰 MiB", fmt(S.evicted_bytes / 1048576, 2)],
          ["旧代提升", S.promotions], ["超大被拒", S.rejected, S.rejected ? "warn" : ""], ["哈希碰撞", S.collisions],
          ["锁竞争", pctf(S.lock_contention), S.lock_contention > 0.02 ? "warn" : ""], ["锁等待 ms", fmt(S.lock_wait_ms, 2)]);
        spark(card.querySelector("canvas"), [push("seg_" + name, S.hit_rate || 0), push("byte_" + name, S.byte_hit_rate || 0)], ["--s0", "--s3"], 1);
        card.querySelector(".bars").innerHTML = S.shard_fill.map((f, i) => `<div title="分片 ${i}: ${pctf(f)}"><i style="height:${Math.min(100, 100 * f)}%;background:var(--s0)"></i></div>`).join("");
      }
      kpis(seg, it);
    }
    const R = T.prefix_cache, pre = card.querySelector(".pre");
    if (!R) kpis(pre, [["状态", "未开启"]]);
    else kpis(pre, [["条目", `${R.entries}/${R.capacity}`], ["查找", R.lookups ?? "–"], ["完整命中", R.exact_hits ?? "–"], ["前缀复用", R.prefix_hits ?? "–"],
      ["未命中", R.misses ?? "–"], ["复用字节占比", pctf(R.reused_share)], ["复用 token", R.reused_tokens ?? "–"]]);
  }
}
async function tick() {
  try {
    const r = await fetch("analyze", {method: "POST", cache: "no-store", body: prev ? JSON.stringify(prev) : ""});
    const j = await r.json();
    if (j.analysis) render(j.analysis, j.snapshot);
    else if (!j.snapshot.process.enabled) render({}, j.snapshot);
    else $("sub").textContent = "已连接，等待第二个快照…";
    prev = j.snapshot;
  } catch (e) { $("sub").innerHTML = `<span class="warn">取数失败：${esc(e)}</span>`; }
}
$("reset").onclick = async () => { await fetch("reset-slow", {method: "POST"}); };
tick(); setInterval(tick, 1000);
</script>
</body>
</html>
"""
