#!/usr/bin/env bash
# followup: extend the §1 corpus to skill_invocation/.
# Runs 3 pods (no_cache, prefix_cache, cacheblend) on the
# single 53-turn skill_invocation capture, sequentially with the
# same queue-patient defaults the rest of the bench uses.

set -euo pipefail

OUT_DIR=benchmark/results/plan6_skill_invocation
mkdir -p "$OUT_DIR"

# Continuous-retry pattern: keep trying every 60s indefinitely (1000 attempts
# = ~17h budget, effectively unlimited for any session we'd actually run).
# Per the capacity-probe-antipattern memory note: don't separately probe
# capacity; the real workload's CREATE call IS the probe. Every retry here
# is a real claim attempt with the full bench payload.
max_attempts=1000
sleep_seconds=60

if [ -f "$OUT_DIR/cacheblend.parquet" ] && \
   [ -f "$OUT_DIR/no_cache.parquet" ] && \
   [ -f "$OUT_DIR/prefix_cache.parquet" ]; then
  echo "=== skill_invocation: all 3 conditions already done — skipping ==="
  exit 0
fi

attempt=1
while [ $attempt -le $max_attempts ]; do
  echo "=== skill_invocation: attempt $attempt/$max_attempts ==="
  if SKILLCACHER_BACKEND_MAX_COMPLETION_TOKENS=512 \
     MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct \
     DTYPE=fp8 \
     GPU_COUNT=2 \
     TENSOR_PARALLEL_SIZE=2 \
     MAX_MODEL_LEN=32768 \
     CLOUD_TYPE=SECURE \
     WAIT_TIMEOUT_S=3600 \
     PROVISIONING_TIMEOUT_S=1800 \
     .venv/bin/skillcacher-bench quality-eval \
       --corpus-dir tests/fixtures/claude_code_real \
       --capture-filter "skill_invocation/plan4_skill_invocation_20260509_102825/_traces" \
       --conditions no_cache,prefix_cache,cacheblend \
       --samples 1 \
       --temperature 0 \
       --out "$OUT_DIR"; then
    echo "=== skill_invocation: done on attempt $attempt ==="
    exit 0
  fi
  echo "=== skill_invocation: attempt $attempt failed; sleeping ${sleep_seconds}s ==="
  sleep $sleep_seconds
  attempt=$((attempt + 1))
done

echo "=== skill_invocation: gave up after $max_attempts attempts ==="
exit 1
