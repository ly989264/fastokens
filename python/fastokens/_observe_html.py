"""The page `fastokens.observe.serve()` serves at `/`: polls `json` once a
second and shows rates over the last window. Self-contained (no network)."""

DASHBOARD = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>fastokens observe</title>
<style>
:root { --bg:#fbfbfa; --fg:#1d1d1b; --muted:#6b6b66; --card:#ffffff; --line:#e4e3de;
        --a:#2f6fdb; --b:#d9822b; --c:#2f9e6b; --d:#b5476b; --bad:#c0392b; }
@media (prefers-color-scheme: dark) {
  :root { --bg:#151514; --fg:#ecebe6; --muted:#9a998f; --card:#1e1e1c; --line:#33322e;
          --a:#6c9cf0; --b:#e9a25a; --c:#5cc293; --d:#e07a9c; --bad:#ef6f61; } }
* { box-sizing:border-box; }
body { margin:0; padding:16px; background:var(--bg); color:var(--fg);
       font:13px/1.45 -apple-system, "PingFang SC", "Segoe UI", sans-serif; }
h1 { font-size:18px; margin:0 0 4px; }
h2 { font-size:14px; margin:0 0 8px; }
.sub { color:var(--muted); margin-bottom:14px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(min(420px, 100%), 1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:12px 14px; overflow-x:auto; min-width:0; }
.kpis { display:flex; flex-wrap:wrap; gap:6px 18px; margin-bottom:8px; }
.kpi b { display:block; font-size:17px; font-variant-numeric:tabular-nums; }
.kpi span { color:var(--muted); font-size:12px; }
table { border-collapse:collapse; width:100%; font-variant-numeric:tabular-nums; }
th, td { text-align:right; padding:3px 6px; border-bottom:1px solid var(--line); white-space:nowrap; }
th:first-child, td:first-child { text-align:left; }
th { color:var(--muted); font-weight:500; }
canvas { width:100%; height:70px; display:block; margin:6px 0; }
.legend { color:var(--muted); font-size:12px; }
.legend i { display:inline-block; width:10px; height:3px; margin:0 4px 2px 10px; vertical-align:middle; }
.warn { color:var(--bad); }
.bars { display:flex; align-items:flex-end; gap:3px; height:80px; margin:6px 0; }
.bars div { flex:1; display:flex; flex-direction:column-reverse; height:100%; background:color-mix(in srgb, var(--line) 50%, transparent); border-radius:2px; overflow:hidden; }
.bars i { display:block; width:100%; }
.note { color:var(--muted); font-size:12px; margin-top:6px; }
</style>
</head>
<body>
<h1>fastokens observe</h1>
<div class="sub" id="sub">连接中…</div>
<div class="grid">
  <div class="card"><h2>概览</h2><div class="kpis" id="overview"></div>
    <canvas id="c_cores"></canvas>
    <div class="legend"><i style="background:var(--a)"></i>调用线程编码（核） <i style="background:var(--b)"></i>线程池任务（核） <i style="background:var(--c)"></i>进程 CPU（核）</div></div>
  <div class="card"><h2>Python API</h2><table id="api"></table>
    <div class="note">p50/p99 取自 log2 直方图，显示所在桶的上界。GIL 释放 = Rust 部分运行时允许其他 Python 线程并行；等回 GIL = Rust 干完后重新拿到 GIL 前等了多久（其他 Python 线程占着 GIL 时变大）。</div></div>
  <div class="card"><h2>编码路径</h2><table id="paths"></table>
    <div class="note">scan_whole：整段扫描；scan_splits：按特殊 token 分段扫描（聊天模板）；segments_cached：经过 segment cache；fused_split / general：非扫描器模型。</div></div>
  <div class="card"><h2>并行 job</h2><div class="kpis" id="pool"></div><table id="jobs"></table>
    <div class="note">单段 ≥64 KB 或整串 ≥64 KB 的聊天输入、超长输入的正则切分、encode_batch 会拆成任务并行；任务在 BPE 线程池（fastokens-N）或 rayon 全局池（rayon-N）上跑。</div></div>
  <div class="card" style="grid-column:1/-1"><h2>线程</h2><table id="threads"></table>
    <div class="note">忙碌 = 窗口内在 fastokens 里干活的时间占比（调用线程按编码计，线程池线程按任务计；任务时间含等待嵌套的并行任务）；API 内 = 调用线程在 fastokens 的 Python API 里的时间占比（含等 GIL、建结果对象）。</div></div>
</div>
<div id="caches"></div>
<script>
const HIST = 120;
let prev = null;
const hist = {};
const $ = id => document.getElementById(id);
const fmt = (v, d=1) => v == null || !isFinite(v) ? "–" : v.toLocaleString(undefined, {maximumFractionDigits:d, minimumFractionDigits:d});
const pctf = v => v == null || !isFinite(v) ? "–" : (100*v).toFixed(1) + "%";
const mib = b => fmt(b / 1048576, 1) + " MiB";
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
function diff(a, b) {
  if (Array.isArray(b)) return b.map((x, i) => typeof x === "number" ? x - ((a && a[i]) || 0) : x);
  if (b && typeof b === "object") { const o = {}; for (const k in b) o[k] = diff(a ? a[k] : undefined, b[k]); return o; }
  if (typeof b === "number") return b - (typeof a === "number" ? a : 0);
  return b;
}
function pct(h, q) { const t = h.reduce((s, x) => s + x, 0); if (!t) return null; let acc = 0;
  for (let i = 0; i < h.length; i++) { acc += h[i]; if (acc >= q * t) return 2 ** i; } return null; }
function kpis(el, items) { el.innerHTML = items.map(([k, v, cls]) => `<div class="kpi"><b class="${cls||""}">${esc(v)}</b><span>${esc(k)}</span></div>`).join(""); }
function table(el, head, rows) { el.innerHTML = "<tr>" + head.map(h => `<th>${esc(h)}</th>`).join("") + "</tr>" +
  rows.map(r => "<tr>" + r.map(c => `<td>${esc(c)}</td>`).join("") + "</tr>").join(""); }
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

function render(p, c) {
  const P = c.process;
  if (!P.enabled) { $("sub").innerHTML = '<span class="warn">这个 fastokens 构建没有开启 observe 特性（maturin build --features observe），没有可显示的计数。</span>'; return; }
  const dt = Math.max(1e-9, P.uptime_s - p.process.uptime_s), D = diff(p.process, P);
  $("sub").textContent = `pid ${c.pid} · 窗口 ${dt.toFixed(1)} s · BPE 线程池 ${P.pool.bpe_threads} 线程 · rayon 全局池 ${P.pool.global_threads} 线程 · 运行 ${fmt(P.uptime_s, 0)} s`;
  const cpu = (c.cpu_s - p.cpu_s) / dt;
  let apiCalls = 0, apiBytes = 0; for (const k in D.api) { apiCalls += D.api[k].calls; if (k.startsWith("encode")) apiBytes += D.api[k].units; }
  let callerBusy = 0, workerBusy = 0; const pt = Object.fromEntries(p.process.threads.map(t => [t.slot, t]));
  const trows = [];
  for (const t of P.threads) { const d = diff(pt[t.slot], t);
    const busy = (t.kind === "worker" ? d.task_ns : d.encode_ns) / dt / 1e9;
    if (t.kind === "worker") workerBusy += busy; else callerBusy += busy;
    if (busy < 0.001 && d.encodes <= 0 && d.tasks <= 0 && d.api_calls <= 0) continue;
    const top = Object.entries(d.paths || {}).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]).map(([k]) => k)[0] || "–";
    trows.push([t.kind === "worker" ? 0 : 1, -busy, [
      (t.name || "slot" + t.slot) + (t.alive ? "" : " (已退出)"), t.kind === "worker" ? "worker" : "调用方", t.os_tid,
      pctf(busy), t.kind === "worker" ? "–" : pctf(d.api_ns / dt / 1e9), fmt(d.encodes / dt, 1), fmt(d.encode_bytes / dt / 1e6, 1),
      top, fmt(d.tasks / dt, 1)]]); }
  trows.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  table($("threads"), ["线程", "类型", "OS tid", "忙碌", "API 内", "编码/s", "MB/s", "主要路径", "任务/s"], trows.map(r => r[2]));
  kpis($("overview"), [["进程 CPU（核）", fmt(cpu, 2)], ["RSS", c.rss_bytes ? mib(c.rss_bytes) : "–"], ["API 调用/s", fmt(apiCalls / dt, 1)],
    ["编码输入 MB/s", fmt(apiBytes / dt / 1e6, 1)], ["调用线程编码（核）", fmt(callerBusy, 2)], ["线程池任务（核）", fmt(workerBusy, 2)]]);
  spark($("c_cores"), [push("caller", callerBusy), push("worker", workerBusy), push("cpu", cpu)], ["--a", "--b", "--c"]);
  table($("api"), ["API", "调用/s", "MB/s", "平均 ms", "p50 ≤ ms", "p99 ≤ ms", "GIL 释放", "等回 GIL ms"],
    Object.entries(D.api).filter(([, o]) => o.calls > 0).map(([k, o]) => [k, fmt(o.calls / dt, 1), fmt(o.units / dt / 1e6, 1),
      fmt(o.ns / o.calls / 1e6, 3), fmt(pct(o.latency_us_log2, .5) / 1e3, 3), fmt(pct(o.latency_us_log2, .99) / 1e3, 3),
      pctf(o.gil_released / o.calls), fmt(o.gil_wait_ns / o.calls / 1e6, 3)]));
  table($("paths"), ["路径", "次/s", "MB/s", "平均 ms", "p99 ≤ ms"],
    Object.entries(D.paths).filter(([, o]) => o.calls > 0).map(([k, o]) => [k, fmt(o.calls / dt, 1), fmt(o.units / dt / 1e6, 1),
      fmt(o.ns / o.calls / 1e6, 3), fmt(pct(o.latency_us_log2, .99) / 1e3, 3)]));
  const lw = D.lock_waits;
  let jobs = 0, tasks = 0; for (const k in D.jobs) { jobs += D.jobs[k].jobs; tasks += D.jobs[k].tasks; }
  kpis($("pool"), [["并行 job/s", fmt(jobs / dt, 1)], ["任务/s", fmt(tasks / dt, 1)], ["每 job 任务数", fmt(jobs ? tasks / jobs : null, 1)],
    ["状态锁读等待/s", fmt(lw.read.count / dt, 1), lw.read.count ? "warn" : ""], ["状态锁写等待/s", fmt(lw.write.count / dt, 1), lw.write.count ? "warn" : ""]]);
  const KN = {scan: "长文本扫描分块", chat_splits: "聊天模板分段", pretokenized: "通用流水线分块", regex_split: "正则切分", batch: "encode_batch 每条"};
  table($("jobs"), ["job 来源", "job/s", "每 job 任务", "平均 ms"], Object.entries(D.jobs).filter(([, v]) => v.jobs > 0)
    .map(([k, v]) => [KN[k] || k, fmt(v.jobs / dt, 1), fmt(v.tasks / v.jobs, 1), fmt(v.ns / v.jobs / 1e6, 3)]));

  const box = $("caches");
  for (const [name, tj] of Object.entries(c.tokenizers)) {
    const C1 = tj.segment_cache; const id = "cache_" + name.replace(/[^A-Za-z0-9_-]/g, "_");
    let card = document.getElementById(id);
    if (!card) { card = document.createElement("div"); card.className = "card"; card.id = id; card.style.marginTop = "12px";
      card.innerHTML = `<h2>Segment cache · ${esc(name)}</h2><div class="kpis"></div><canvas></canvas>
        <div class="legend"><i style="background:var(--a)"></i>分段命中率 <i style="background:var(--b)"></i>字节命中率</div>
        <div class="legend" style="margin-top:8px">各分片占用（下：old 代，上：young 代；满格 = 该分片预算）</div><div class="bars"></div>
        <div class="grid" style="margin-top:8px"><div><div class="legend">命中分段大小分布（累计，每 16 次查找抽样 1 次）</div><table class="hh"></table></div>
        <div><div class="legend">未命中分段大小分布（累计，每 16 次查找抽样 1 次）</div><table class="mh"></table></div></div>`; box.appendChild(card); }
    const k = card.querySelector(".kpis");
    if (!C1) { kpis(k, [["状态", "未开启（FASTOKENS_SEGMENT_CACHE）"], ["扫描器快路径", tj.scanner_path ? "是" : "否"]]); continue; }
    const C0 = (p.tokenizers[name] || {}).segment_cache || {}, dc = diff(C0, C1);
    const items = [["占用", `${mib(C1.bytes)} / ~${mib(C1.approx_max_bytes)}`], ["条目", C1.entries.toLocaleString()],
      ["单段上限", fmt(C1.max_segment_cost / 1024, 1) + " KB 成本"]];
    if (C1.shards) {
      const segHit = dc.hits / Math.max(1, dc.hits + dc.misses), byteHit = dc.hit_bytes / Math.max(1, dc.hit_bytes + dc.miss_bytes);
      const sh = C1.shards.map((s, i) => diff((C0.shards || [])[i], s)), sum = key => sh.reduce((a, s) => a + s[key], 0);
      const acq = dc.hits + dc.misses + sum("inserts");
      items.push(["分段命中率（窗口）", pctf(segHit)], ["字节命中率（窗口）", pctf(byteHit)],
        ["代际轮换/s", fmt(sum("rotations") / dt, 2)], ["淘汰 MiB/s", fmt(sum("evicted_bytes") / dt / 1048576, 2)], ["旧代提升/s", fmt(sum("promotions") / dt, 1)],
        ["插入/s", fmt(sum("inserts") / dt, 1)], ["超大被拒/s", fmt(dc.rejected_too_large / dt, 2), dc.rejected_too_large > 0 ? "warn" : ""],
        ["哈希碰撞", C1.shards.reduce((a, s) => a + s.collisions, 0)], ["锁竞争", pctf(sum("lock_contended") / Math.max(1, acq)), sum("lock_contended") / Math.max(1, acq) > 0.02 ? "warn" : ""],
        ["锁等待 ms/s", fmt(sum("lock_wait_ns") / dt / 1e6, 2)], ["未命中分词 ms/s", fmt(dc.miss_tokenize_ns / dt / 1e6, 1)]);
      const per = 2 * C1.young_budget_per_shard;
      card.querySelector(".bars").innerHTML = C1.shards.map((s, i) => `<div title="分片 ${i}: young ${mib(s.young_bytes)} / old ${mib(s.old_bytes)}">
        <i style="height:${Math.min(100, 100 * s.old_bytes / per)}%;background:var(--b)"></i><i style="height:${Math.min(100, 100 * s.young_bytes / per)}%;background:var(--a)"></i></div>`).join("");
      const human = b => b >= 1048576 ? (b / 1048576) + " MB" : b >= 1024 ? (b / 1024) + " KB" : b + " B";
      const sizes = h => h.map((n, i) => [i ? `${human(2 ** (i - 1))} – ${human(2 ** i)}` : "0 B", n.toLocaleString()]).filter(r => r[1] !== "0");
      table(card.querySelector(".hh"), ["大小", "次数"], sizes(C1.hit_sizes_log2));
      table(card.querySelector(".mh"), ["大小", "次数"], sizes(C1.miss_sizes_log2));
      spark(card.querySelector("canvas"), [push("seg_" + name, segHit), push("byte_" + name, byteHit)], ["--a", "--b"], 1);
    }
    kpis(k, items);
  }
}
async function tick() {
  try { const c = await (await fetch("json", {cache: "no-store"})).json(); if (prev) render(prev, c); else $("sub").textContent = "已连接，等待第二个快照…"; prev = c; }
  catch (e) { $("sub").innerHTML = `<span class="warn">取数失败：${esc(e)}</span>`; }
}
tick(); setInterval(tick, 1000);
</script>
</body>
</html>
"""
