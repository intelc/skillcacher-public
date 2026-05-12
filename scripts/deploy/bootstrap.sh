#!/usr/bin/env bash
# Run ON the GPU box after provision.sh brings it up.
# Idempotent: re-running on a warm box is a no-op for steps already done.
set -euo pipefail

ENV_FILE="${ENV_FILE:-$(dirname "$0")/../../.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi

: "${TAILSCALE_AUTH_KEY:?TAILSCALE_AUTH_KEY required}"
MODEL_NAME="${MODEL_NAME:-meta-llama/Llama-3.3-70B-Instruct}"
MODEL_DIR="${MODEL_DIR:-/models/llama-3.3-70b}"
VLLM_PORT="${VLLM_PORT:-8000}"
HOSTNAME_TAG="${POD_NAME:-skillcacher-bench}"

echo "[bootstrap] step 1: tailscale"
if ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi
tailscale up --auth-key "$TAILSCALE_AUTH_KEY" --hostname "$HOSTNAME_TAG" --ssh
TS_IP=$(tailscale ip -4)
echo "[bootstrap] tailscale ip: $TS_IP"

echo "[bootstrap] step 2: pinned python deps"
if ! python3 -c "import vllm" 2>/dev/null; then
  echo "[bootstrap] FATAL: vllm not on PATH — wrong template/image?"
  exit 1
fi
# LMCache version policy:
#   - If requirements-bench.txt pins it (lmcache==X), enforce that version.
#   - Otherwise, accept whatever the image baked in (preferred path with
#     lmcache/vllm-openai). FATAL only if neither source provides it.
LMCACHE_PIN="$(grep -E '^lmcache==' requirements-bench.txt | head -1)"
if [[ -n "$LMCACHE_PIN" ]]; then
  if ! python3 -c "import lmcache; assert lmcache.__version__ == '${LMCACHE_PIN##*==}'" 2>/dev/null; then
    echo "[bootstrap] installing $LMCACHE_PIN..."
    pip install "$LMCACHE_PIN"
  fi
else
  if ! BAKED_VERSION="$(python3 -c 'import lmcache; print(lmcache.__version__)' 2>/dev/null)"; then
    echo "[bootstrap] FATAL: no lmcache available — image lacks it and requirements-bench.txt has no pin"
    exit 1
  fi
  echo "[bootstrap] using image-baked lmcache $BAKED_VERSION (no pin in requirements-bench.txt)"
fi
pip install --upgrade pip
pip install -r requirements-bench.txt

echo "[bootstrap] step 3: huggingface auth + model download"
: "${HF_TOKEN:?HF_TOKEN required for Llama-3.3 download}"
mkdir -p "$MODEL_DIR"
# HF_TOKEN read from os.environ inside python so it doesn't leak into
# /proc/<pid>/cmdline (visible to any process via `ps aux`).
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
  HF_TOKEN="$HF_TOKEN" MODEL_NAME="$MODEL_NAME" MODEL_DIR="$MODEL_DIR" python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download(os.environ['MODEL_NAME'], local_dir=os.environ['MODEL_DIR'], token=os.environ['HF_TOKEN'], allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
"
fi

# Also: pgrep -f below uses `vllm.entrypoints\|vllm serve` — those substrings
# don't appear in this bootstrap.sh's PID 1 cmdline, so they're safe here.
# In dockerStartCmd-baked flows (oneshot_pod.py) prefer `pgrep -x vllm` to
# avoid matching the giant PID-1 bash command.

echo "[bootstrap] step 4: launch vLLM (idempotent)"
if pgrep -f "vllm.entrypoints\|vllm serve" >/dev/null; then
  echo "[bootstrap] vLLM already running, skipping spawn"
else
  # CONDITION env-var controls cache mode — set by provision.sh per an upstream task.
  CONDITION="${CONDITION:-prefix_cache}"
  case "$CONDITION" in
    no_cache)
      unset VLLM_KV_CACHE_TYPE LMCACHE_LOCAL_CPU LMCACHE_USE_EXPERIMENTAL
      export LMCACHE_DISABLE=1
      echo "[bootstrap] CONDITION=no_cache (LMCache disabled)"
      ;;
    prefix_cache)
      export VLLM_KV_CACHE_TYPE=lmcache
      export LMCACHE_USE_EXPERIMENTAL=False
      export LMCACHE_LOCAL_CPU=True
      export LMCACHE_MAX_LOCAL_CPU_SIZE=40
      export LMCACHE_CHUNK_SIZE=256
      unset LMCACHE_ENABLE_BLENDING
      echo "[bootstrap] CONDITION=prefix_cache (LMCache prefix-only)"
      ;;
    cacheblend)
      # CacheBlend requires layerwise execution + blend-specific configs
      # per LMCache docs (kv_cache_optimizations/blending.html). Without
      # LMCACHE_USE_LAYERWISE=True, vllm crashes at startup.
      export VLLM_KV_CACHE_TYPE=lmcache
      export LMCACHE_USE_EXPERIMENTAL=True
      export LMCACHE_LOCAL_CPU=True
      export LMCACHE_MAX_LOCAL_CPU_SIZE=40
      export LMCACHE_CHUNK_SIZE=256
      export LMCACHE_ENABLE_BLENDING=True
      export LMCACHE_USE_LAYERWISE=True
      export LMCACHE_BLEND_SPECIAL_STR=" # # "
      export LMCACHE_BLEND_CHECK_LAYERS=1
      export LMCACHE_BLEND_RECOMPUTE_RATIOS=0.15
      echo "[bootstrap] CONDITION=cacheblend (LMCache + CacheBlend)"
      ;;
    *)
      echo "[bootstrap] FATAL: unknown CONDITION=$CONDITION"
      exit 1
      ;;
  esac

  # Bind to 0.0.0.0 so RunPod's public proxy URL works alongside Tailscale.
  # Single-machine isolation; security boundary is the api-key + tailnet ACL.
  # tool-call-parser flags enable structured tool-call parsing for capture
  # fidelity (Qwen3-8B + Llama-3.3 are both hermes-compatible).
  nohup vllm serve "$MODEL_DIR" \
    --served-model-name "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --dtype fp8 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 64 \
    --enable-auto-tool-choice \
    --tool-call-parser hermes \
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}' \
    >/var/log/vllm.log 2>&1 &
fi

echo "[bootstrap] step 5: launch lmcache shim"
if pgrep -f "lmcache_shim" >/dev/null; then
  echo "[bootstrap] lmcache shim already running"
else
  : "${LMCACHE_SHIM_API_KEY:?required for shim auth}"
  nohup python3 -m scripts.deploy.lmcache_shim >/var/log/lmcache_shim.log 2>&1 &
  echo "[bootstrap] lmcache shim on $TS_IP:8001"
fi

echo "[bootstrap] vLLM launching on 0.0.0.0:$VLLM_PORT (logs: /var/log/vllm.log)"
echo "[bootstrap] reachable two ways:"
echo "  - Tailscale path: http://$TS_IP:$VLLM_PORT"
echo "  - RunPod public proxy: https://<pod_id>-$VLLM_PORT.proxy.runpod.net"
