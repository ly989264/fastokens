#!/bin/bash
# Full functional test of this checkout on Linux (no performance tests).
# usage: lt/run.sh OUTDIR   (env: PY=python3.12, SHAREGPT=<json> or unset to
#        synthesize, FUZZ_SECS=30, STRESS=<extra stress script> optional)
set -u
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$1; mkdir -p "$OUT"
PY=${PY:-python3}
FUZZ_SECS=${FUZZ_SECS:-30}
SUM=$OUT/summary.txt; : > "$SUM"
cd "$REPO"
export CARGO_TERM_COLOR=never
note() { echo "$*" | tee -a "$SUM"; }
step() { # name logfile command...
  local name=$1 log=$OUT/$2; shift 2
  local t0=$(date +%s)
  "$@" > "$log" 2>&1; local rc=$?
  note "[$( [ $rc -eq 0 ] && echo PASS || echo FAIL )] $name (rc=$rc, $(( $(date +%s) - t0 )) s, log $(basename "$log"))"
  return $rc
}
tests_line() { # summarize cargo test results in a log
  awk '/^test result:/ {p+=$4; f+=$6; i+=$8} END {printf "    cargo tests: %d passed, %d failed, %d ignored\n", p, f, i}' "$1" | tee -a "$SUM"
  grep -E "^test .* FAILED$|^    [a-z_:]+$" "$1" | head -20 | sed 's/^/    /' | tee -a "$SUM"
}
note "== $(uname -srm); $($PY --version 2>&1); $(rustc --version); commit $(git rev-parse --short HEAD)"

# 1. Rust
step "cargo fmt --check" fmt.log cargo fmt --check
step "cargo clippy -- -D warnings" clippy.log cargo clippy -- -D warnings
step "cargo clippy --features observe -- -D warnings" clippy-obs.log cargo clippy --features observe -- -D warnings
step "cargo test --release" test.log cargo test --release; tests_line "$OUT/test.log"
step "cargo test --release --features observe" test-obs.log cargo test --release --features observe; tests_line "$OUT/test-obs.log"
for repo in zai-org/GLM-5.3 deepseek-ai/DeepSeek-V4.1-Flash Qwen/Qwen3-Coder-30B-A3B-Instruct; do
  tj=$($PY -c "from huggingface_hub import hf_hub_download as d; print(d('$repo', 'tokenizer.json'))" 2>/dev/null)
  n=$(basename "$repo")
  if [ -z "$tj" ]; then note "[SKIP] prefix fuzz $n (tokenizer.json not available)"; continue; fi
  for strip in 0 1; do
    step "prefix fuzz $n strip=$strip ${FUZZ_SECS}s" "fuzz-$n-$strip.log" \
      env FUZZ_TOKENIZER="$tj" FUZZ_SECS="$FUZZ_SECS" FUZZ_SEED=1 FUZZ_STRIP=$strip \
      cargo test --release --test prefix_cache_fuzz -- --ignored --nocapture
    grep -iE "requests|mismatch|compared" "$OUT/fuzz-$n-$strip.log" | tail -2 | sed 's/^/    /' | tee -a "$SUM"
  done
done

# 2. wheels and pytest
DEPS=("pytest==9.1.1" "tokenizers==0.22.2" "transformers==5.12.1" "tiktoken==0.14.0" "huggingface-hub==1.33.0" numpy psutil jinja2 "maturin>=1,<2")
for v in default observe; do
  feat=""; [ $v = observe ] && feat="--features observe"
  $PY -m venv "$OUT/venv-$v" && "$OUT/venv-$v/bin/pip" install -q "${DEPS[@]}" > "$OUT/pip-$v.log" 2>&1
  CARGO_TARGET_DIR=$REPO/target/py-$v step "maturin build ($v)" "wheel-$v.log" \
    "$OUT/venv-$v/bin/maturin" build --release $feat --out "$OUT/dist-$v"
  "$OUT/venv-$v/bin/pip" install -q --force-reinstall --no-deps "$OUT"/dist-$v/fastokens-*.whl >> "$OUT/pip-$v.log" 2>&1
  note "    installed: $("$OUT/venv-$v/bin/python" -c "import fastokens.observe as o; print('observe' if o.enabled() else 'default')")"
  step "pytest python/tests ($v)" "pytest-$v.log" "$OUT/venv-$v/bin/python" -m pytest python/tests -q -rf -p no:cacheprovider
  tail -1 "$OUT/pytest-$v.log" | sed 's/^/    /' | tee -a "$SUM"
  grep "^FAILED" "$OUT/pytest-$v.log" | sed 's/^/    /' | tee -a "$SUM"
done

# 3. cache consistency on real models (vs HF tokenizers / tiktoken)
V=$OUT/venv-default/bin/python
step "check.py all models --threads 16" check-t16.log $V examples/agent/check.py --threads 16
grep -E "^  [a-z0-9.-]+: |FAIL|ERROR|ALL CHECKS" "$OUT/check-t16.log" | grep -v " ok " | sed 's/^/    /' | head -20 | tee -a "$SUM"
for t in 1 2 8; do step "check.py --quick --threads $t" "check-q$t.log" $V examples/agent/check.py --quick --threads $t; done
step "check.py --quick --threads 8 (observe wheel)" check-obs.log "$OUT/venv-observe/bin/python" examples/agent/check.py --quick --threads 8
if [ -z "${SHAREGPT:-}" ]; then SHAREGPT=$OUT/sharegpt-synth.json; $PY lt/synth_sharegpt.py "$REPO" "$SHAREGPT"; fi
step "prefix_check.py all models" prefix-check.log $V examples/agent/prefix_check.py --sharegpt "$SHAREGPT" --agent 60 --chat 60
grep -E "ERROR|FAIL|mismatch|PASSED|passed|failed" "$OUT/prefix-check.log" | tail -20 | sed 's/^/    /' | tee -a "$SUM"
if [ -n "${STRESS:-}" ] && [ -f "$STRESS" ]; then
  cp "$STRESS" examples/agent/_stress.py
  step "stress threads mode (1-16 threads, both caches)" stress-threads.log $V examples/agent/_stress.py threads --input-cache 64
  rm -f examples/agent/_stress.py
  tail -5 "$OUT/stress-threads.log" | sed 's/^/    /' | tee -a "$SUM"
fi

# 4. observe tools, 5. vLLM plugin (no vLLM)
for v in default observe; do
  step "observe tools ($v)" "tools-$v.log" "$OUT/venv-$v/bin/python" lt/tools_check.py "$REPO" "$OUT/tools-$v"
  grep -E "^FAIL|^INFO|^TOOLS" "$OUT/tools-$v.log" | sed 's/^/    /' | tee -a "$SUM"
done
step "vllm_observe plugin (no vLLM)" plugin.log env FASTOKENS_SEGMENT_CACHE=16 "$OUT/venv-observe/bin/python" lt/plugin_check.py "$REPO"
grep -E "^FAIL|^PLUGIN" "$OUT/plugin.log" | sed 's/^/    /' | tee -a "$SUM"
note "== done"
