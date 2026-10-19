"""Check every observe tool against what docs/OPERATIONS.zh-CN.md says for the
installed variant (default or observe). Prints one PASS/FAIL line per check
and a final count. usage: python lt/tools_check.py REPO OUTDIR"""
import glob, json, os, subprocess, sys, threading, time, urllib.request

REPO, OUT = sys.argv[1], sys.argv[2]
PY = sys.executable
os.makedirs(OUT, exist_ok=True)
from huggingface_hub import hf_hub_download  # noqa: E402

TJ = hf_hub_download("zai-org/GLM-5.2", "tokenizer.json")
import fastokens.observe as _o  # noqa: E402

OBS = _o.enabled()
results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}"[:400], flush=True)


def run(args, env=None, timeout=300):
    r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env={**os.environ, **(env or {})})
    return r.returncode, r.stdout, r.stderr


NOTHING = "nothing to show"
rc, out, _ = run([PY, "-c", "import fastokens.observe as o; print('observe' if o.enabled() else 'default')"])
check("variant", rc == 0, out.strip())

# a process with both caches, serving observe, under load
demo = f'''
import os, sys, threading, time
os.environ["FASTOKENS_SEGMENT_CACHE"] = "256"; os.environ["FASTOKENS_INPUT_CACHE"] = "16"
import fastokens; fastokens.patch_transformers()
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained({os.path.dirname(TJ)!r})
import fastokens.observe as obs
msgs = [{{"role": "system", "content": "You are a coding agent. " * 300}}]; reqs = []
for t in range(12):
    msgs.append({{"role": "user", "content": ("read src/lib.rs and fix the failing test %d " % t) * 120}})
    reqs.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
    msgs.append({{"role": "assistant", "content": ("done step %d " % t) * 80}})
server = obs.serve({{"glm": tok}}, port=0)
print("PORT", server.server_address[1], flush=True)
a = obs.snapshot({{"glm": tok}})
with obs.profile({{"glm": tok}}) as p:
    for r in reqs: tok(r)["input_ids"]
print("PROFILE", repr(p.report()[:300]), flush=True)
print("REPORT", repr(obs.report(a, obs.snapshot({{"glm": tok}}))[:300]), flush=True)
end = time.time() + float(os.environ.get("HOLD", "40"))
while time.time() < end:
    for r in reqs: tok(r)["input_ids"]
'''
open(f"{OUT}/demo.py", "w").write(demo)
p = subprocess.Popen([PY, f"{OUT}/demo.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env={**os.environ, "HOLD": "45"})
lines = {}
for line in p.stdout:
    k, _, v = line.partition(" ")
    lines[k] = v.strip()
    if k == "REPORT":
        break
url = f"http://127.0.0.1:{lines['PORT']}"
get = lambda path: urllib.request.urlopen(url + path, timeout=30).read().decode()  # noqa: E731
check("profile() block", (NOTHING in lines["PROFILE"]) != OBS, lines["PROFILE"][:120])
check("report()", (NOTHING in lines["REPORT"]) != OBS, lines["REPORT"][:120])
html = get("/")
check("serve /", "<html" in html.lower() and len(html) > 5000, f"{len(html)} bytes")
snap = json.loads(get("/json"))
seg = snap["tokenizers"]["glm"]["segment_cache"]; chat = snap["tokenizers"]["glm"]["prefix_cache"]["chat"]
check("serve /json", set(snap) >= {"pid", "process", "tokenizers"} and seg["entries"] > 0 and chat["entries"] > 0,
      f"segment entries {seg['entries']}, prefix entries {chat['entries']}")
check("hit counters only in observe build", ("hits" in seg and "lookups" in chat) == OBS,
      f"segment hits {seg.get('hits')}, prefix lookups {chat.get('lookups')}")
m = get("/metrics")
check("serve /metrics", m.startswith("#") and (("fastokens_segment_cache_hits_total" in m) == OBS), f"{len(m.splitlines())} lines")
r = urllib.request.urlopen(urllib.request.Request(url + "/reset-slow", data=b"", method="POST")).read()
check("serve POST /reset-slow", r == b"{}", repr(r))
rc, out, err = run([PY, "-m", "fastokens.observe", "top", "--once", "--url", url])
check("top --once", rc == 0 and (NOTHING in out) != OBS and (not OBS or "segment cache [glm]" in out), out.splitlines()[0][:100] if out else err[-200:])
rc, out, err = run([PY, "-m", "fastokens.observe", "profile", "--url", url, "--seconds", "3", "--slowest", "2"])
check("profile --seconds 3", rc == 0 and (NOTHING in out) != OBS and (not OBS or "encodes" in out), out.splitlines()[1][:100] if OBS and out else out[:100])
rc, out, err = run([PY, "-m", "fastokens.observe", "dump", "--url", url])
check("dump", rc == 0 and "tokenizers" in json.loads(out), f"{len(out)} bytes")
rc, out, err = run([PY, "-m", "fastokens.observe", "sample", "--pid", str(p.pid), "--seconds", "4", "--out", f"{OUT}/pid"])
check("sample --pid", rc == 0 and "where the CPU goes" in out, (out.splitlines() or [err[-300:]])[0])
p.kill(); p.wait()
files = sorted(os.path.basename(f) for f in glob.glob(f"{OUT}/pid.*"))
check("sample --out files", files == ["pid.folded", "pid.json", "pid.raw.txt", "pid.txt"], str(files))
rc, out, err = run([PY, "-m", "fastokens.observe", "sample", "--from", f"{OUT}/pid.raw.txt"])
check("sample --from", rc == 0 and "where the CPU goes" in out, (out.splitlines() or [err[-200:]])[0])
loop = f"import fastokens\nt = fastokens.Tokenizer.from_file({TJ!r})\nx = open({TJ!r}).read()[:200000]\nwhile True: t.encode(x).ids\n"
open(f"{OUT}/loop.py", "w").write(loop)
rc, out, err = run([PY, "-m", "fastokens.observe", "sample", "--seconds", "4", "--delay", "1", "--out", f"{OUT}/cmd", "--", PY, f"{OUT}/loop.py"])
check("sample -- command", rc == 0 and "where the CPU goes" in out, (out.splitlines() or [err[-300:]])[0])
if rc == 0:
    s = json.load(open(f"{OUT}/cmd.json"))
    other = s["categories_on_cpu"].get("other", {}).get("share", 0)
    jit = sum(t["share_on_cpu"] for t in s["top_self"] if t["function"] == "[unknown]" and t["lib"].startswith("perf-"))
    regex = s["categories_on_cpu"].get("regex engine", {}).get("share", 0)
    print(f"INFO  sampler output on this loop: other {100 * other:.1f}%, [unknown] in perf-<pid>.map {100 * jit:.1f}%, "
          f"regex engine {100 * regex:.1f}%, on-CPU samples {s['on_cpu_samples']}")
json.dump([open(TJ).read()[i:i + 30000] for i in range(0, 300000, 30000)], open(f"{OUT}/prompts.json", "w"))
rc, out, err = run([PY, "-m", "fastokens.observe", "layers", "--tokenizer", TJ, "--inputs", f"{OUT}/prompts.json", "--calls", "50"])
check("layers native", (rc == 0 and "fastokens Rust" in out) if OBS else (rc != 0 and "observe" in err), (out.splitlines() or err.strip().splitlines()[-1:] or [""])[0][:120])
rc, out, err = run([PY, "-m", "fastokens.observe", "layers", "--tokenizer", os.path.dirname(TJ), "--via", "transformers", "--inputs", f"{OUT}/prompts.json", "--calls", "30"])
check("layers --via transformers", (rc == 0 and "fastokens Rust" in out) if OBS else (rc != 0 and "observe" in err), (out.splitlines() or err.strip().splitlines()[-1:] or [""])[0][:120])
print(f"TOOLS {'observe' if OBS else 'default'}: {sum(results)} passed, {len(results) - sum(results)} failed")
