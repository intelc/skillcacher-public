#!/usr/bin/env bash
# Capture real Claude Code traffic on SWE-Bench Lite tasks.
# Stores raw traces under tests/fixtures/claude_code_real/<task_id>/.
# Requires: a `prefix_cache` pod up; ANTHROPIC_BASE_URL pointed at proxy.
set -euo pipefail

ENV_FILE="${ENV_FILE:-$(dirname "$0")/../.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi

: "${SKILLCACHER_BACKEND_URL:?must be set to the pod URL}"
: "${ANTHROPIC_BASE_URL:?must point at the local skillcacher proxy}"

TASKS_FILE="${TASKS_FILE:-scripts/swebench_lite_tasks.txt}"
OUT_ROOT="${OUT_ROOT:-tests/fixtures/claude_code_real}"
TRACE_DIR_BASE="${TRACE_DIR_BASE:-benchmark/capture_traces}"

mkdir -p "$OUT_ROOT" "$TRACE_DIR_BASE"

while IFS= read -r TASK; do
    [[ -z "$TASK" || "$TASK" =~ ^# ]] && continue
    echo "[capture] $TASK"
    TASK_TRACE_DIR="$TRACE_DIR_BASE/$TASK"
    mkdir -p "$TASK_TRACE_DIR"
    SKILLCACHER_TRACE_DIR="$TASK_TRACE_DIR" timeout 1200 \
        claude --task "Fix SWE-Bench task $TASK" --max-turns 50 || true
    cp -r "$TASK_TRACE_DIR" "$OUT_ROOT/$TASK"
done < "$TASKS_FILE"

echo "[capture] done. Run: python scripts/redact.py $OUT_ROOT --in-place"
