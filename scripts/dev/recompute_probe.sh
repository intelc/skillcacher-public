#!/usr/bin/env bash
# LMCACHE_BLEND_RECOMPUTE_RATIOS sweep.
#
# Runs 3 cacheblend pods sequentially on the pytest-7490 fixture at
# ratios {0.15, 0.5, 1.0}. Each pod replays the 4 captured turns at
# samples=1, T=0, matching §1 corpus methodology.
#
# Reuses the queue-patient defaults from runpod_capacity_retry memory:
# CLOUD_TYPE=SECURE + WAIT_TIMEOUT_S=3600 + PROVISIONING_TIMEOUT_S=1800.

set -euo pipefail

OUT_ROOT=benchmark/results/plan5_recompute_probe

run_ratio() {
  local ratio=$1
  local label=$2
  local out_dir="$OUT_ROOT/$label"
  if [ -f "$out_dir/cacheblend.parquet" ]; then
    echo "=== [r=$ratio] already done at $out_dir — skipping ==="
    return 0
  fi
  mkdir -p "$out_dir"
  local max_attempts=20
  local sleep_seconds=120
  local attempt=1
  while [ $attempt -le $max_attempts ]; do
    echo "=== [r=$ratio] attempt $attempt/$max_attempts, out=$out_dir ==="
    if SKILLCACHER_BACKEND_MAX_COMPLETION_TOKENS=512 \
       MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
       DTYPE=fp8 \
       GPU_COUNT=2 \
       TENSOR_PARALLEL_SIZE=2 \
       MAX_MODEL_LEN=32768 \
       CLOUD_TYPE=SECURE \
       WAIT_TIMEOUT_S=3600 \
       PROVISIONING_TIMEOUT_S=1800 \
       LMCACHE_BLEND_RECOMPUTE_RATIOS_OVERRIDE="$ratio" \
       .venv/bin/skillcacher-bench quality-eval \
         --corpus-dir tests/fixtures/claude_code_real \
         --capture-filter "swebench_verified/pytest-dev__pytest-7490" \
         --conditions cacheblend \
         --samples 1 \
         --temperature 0 \
         --out "$out_dir"; then
      echo "=== [r=$ratio] done on attempt $attempt ==="
      return 0
    fi
    echo "=== [r=$ratio] attempt $attempt failed; sleeping ${sleep_seconds}s before retry ==="
    sleep $sleep_seconds
    attempt=$((attempt + 1))
  done
  echo "=== [r=$ratio] gave up after $max_attempts attempts ==="
  return 1
}

run_ratio 0.15 r015
run_ratio 0.5  r05
run_ratio 1.0  r10

echo "=== sweep complete ==="
