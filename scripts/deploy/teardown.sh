#!/usr/bin/env bash
# Stop the GPU box. Keeps the persistent network volume so weights survive.
# --keep-volume is currently a no-op (volume is always retained); plumbed
# through for forward-compat and so callers passing the flag don't error.
set -euo pipefail

ENV_FILE="${ENV_FILE:-$(dirname "$0")/../../.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi

: "${RUNPOD_API_KEY:?RUNPOD_API_KEY required}"

KEEP_VOLUME=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-volume) KEEP_VOLUME=1; shift ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

export KEEP_VOLUME
python3 "$(dirname "$0")/_teardown.py"
