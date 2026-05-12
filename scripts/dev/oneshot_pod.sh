#!/usr/bin/env bash
# Create or resume a self-bootstrapping vLLM+lmcache pod for dev/validation.
# Wraps scripts/dev/oneshot_pod.py; on success, exports POD_ID and PROXY_URL.
#
# Usage:
#   bash scripts/dev/oneshot_pod.sh
#   source scripts/dev/oneshot_pod.sh    # makes POD_ID + PROXY_URL available
#
# Reads .env automatically (the python script handles it). Optional env:
#   POD_NAME (default: skillcacher-dev-prefix_cache)
#   IMAGE    (default: vllm/vllm-openai:latest)
#   MODEL_NAME (default: Qwen/Qwen3-8B)
#   WAIT_TIMEOUT_S (default: 900 — accommodate first-time model download)
#
# Teardown: bash scripts/deploy/teardown.sh   (uses POD_NAME to find the pod;
# set POD_NAME explicitly if you used a non-default).

set -euo pipefail

OUT="$(python3 "$(dirname "$0")/oneshot_pod.py")"
echo "$OUT"
export POD_ID="$(echo "$OUT" | sed -n 's/^POD_ID=//p')"
export PROXY_URL="$(echo "$OUT" | sed -n 's/^PROXY_URL=//p')"

if [[ -z "$POD_ID" || -z "$PROXY_URL" ]]; then
    echo "[oneshot.sh] FATAL: missing POD_ID or PROXY_URL" >&2
    exit 1
fi
echo "[oneshot.sh] ready: POD_ID=$POD_ID PROXY_URL=$PROXY_URL"
