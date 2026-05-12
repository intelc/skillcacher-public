#!/usr/bin/env bash
# Provision a RunPod pod for skillcacher bench.
# Usage: provision.sh [--condition no_cache|prefix_cache|cacheblend]
set -euo pipefail

ENV_FILE="${ENV_FILE:-$(dirname "$0")/../../.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi

CONDITION="${CONDITION:-prefix_cache}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --condition) CONDITION="$2"; shift 2 ;;
    *) echo "unknown arg: $1"; exit 1 ;;
  esac
done

case "$CONDITION" in
  no_cache|prefix_cache|cacheblend) ;;
  *) echo "FATAL: --condition must be no_cache|prefix_cache|cacheblend"; exit 1 ;;
esac

export CONDITION
echo "[provision] CONDITION=$CONDITION"

# POD_NAME inherits the condition so different conditions don't collide
export POD_NAME="${POD_NAME:-skillcacher-bench}-${CONDITION}"

python3 "$(dirname "$0")/_provision.py"
