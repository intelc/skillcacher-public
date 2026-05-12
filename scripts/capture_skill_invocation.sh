#!/usr/bin/env bash
# capture (skill, prompt) pairs that explicitly invoke a hand-
# crafted SKILL.md anchor, used to demonstrate skill_hit_rate > 0 on the
# cacheblend condition.
#
# Per-task flow:
# 1. Read the prompt from $PROMPT_DIR/$task_id.txt (built by
#    `python -m scripts.skill_invocation_prompts`).
# 2. Run `claude --bare -p` against ANTHROPIC_BASE_URL=<proxy>. The proxy
#    parses the verbatim skill anchor in the prompt, looks it up in the
#    SkillPrefixIndex, and (under cacheblend) routes through the
#    pre-seeded lmcache for retrieval.
# 3. Stdout per task lands in $RAW_ROOT/$task_id/_claude_stdout.txt.
# 4. Per-request token parquets land in $SKILLCACHER_TRACE_DIR (a single
#    shared dir for the whole batch — the proxy reads it once at spawn).
# 5. Redact each task's stdout dir in place after the loop.
#
# The pod stays warm across all tasks. vllm.log is dumped by the
# orchestrator on teardown to $RAW_ROOT/vllm.log; the per-request
# `LMCache hit tokens: N` lines correlate to request_id which is also
# stamped on each parquet, so we can post-hoc bucket hits per task.
#
# Requires:
# - $TASKS_FILE: one task_id per line (the file produced alongside the
#   per-task prompt files).
# - $PROMPT_DIR: contains $task_id.txt per task.
# - $RAW_ROOT: parent directory for per-task stdout subdirs.
# - ANTHROPIC_BASE_URL: pointed at the local skillcacher proxy.
set -euo pipefail

ENV_FILE="${ENV_FILE:-$(dirname "$0")/../.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi

: "${ANTHROPIC_BASE_URL:?must point at the local skillcacher proxy}"
: "${TASKS_FILE:?must be set to a file of task_id lines}"
: "${PROMPT_DIR:?must be set to a directory of <task_id>.txt prompts}"
: "${RAW_ROOT:?must be set to the per-batch raw output root}"

TIMEOUT_SECS="${TIMEOUT_SECS:-600}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-0.5}"

# Portable timeout / python-bin helpers (mirrored from capture_long_sessions.sh).
if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="gtimeout"
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout"
else
    TIMEOUT_BIN=""
    echo "[capture-skill] note: no timeout binary; relying on --max-budget-usd as soft cap" >&2
fi
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

mkdir -p "$RAW_ROOT"

# `--bare` mode skips OAuth/keychain; the proxy doesn't auth-check, so any
# non-empty key passes through. Mirrors capture_long_sessions.sh.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-EMPTY}"

n_total=0
n_done=0

while IFS= read -r TASK; do
    [[ -z "$TASK" || "$TASK" =~ ^# ]] && continue
    n_total=$((n_total + 1))

    PROMPT_FILE="$PROMPT_DIR/${TASK}.txt"
    if [[ ! -f "$PROMPT_FILE" ]]; then
        echo "[capture-skill] $TASK: prompt file $PROMPT_FILE missing; skipping" >&2
        continue
    fi
    TASK_RAW_DIR="$RAW_ROOT/$TASK"
    mkdir -p "$TASK_RAW_DIR"

    echo "[capture-skill] $TASK (timeout=${TIMEOUT_SECS}s, budget=\$${MAX_BUDGET_USD})"

    if [[ -n "$TIMEOUT_BIN" ]]; then
        "$TIMEOUT_BIN" "$TIMEOUT_SECS" \
            claude --bare -p "$(cat "$PROMPT_FILE")" \
                --max-budget-usd "$MAX_BUDGET_USD" \
                --dangerously-skip-permissions \
                > "$TASK_RAW_DIR/_claude_stdout.txt" 2>&1 || true
    else
        claude --bare -p "$(cat "$PROMPT_FILE")" \
            --max-budget-usd "$MAX_BUDGET_USD" \
            --dangerously-skip-permissions \
            > "$TASK_RAW_DIR/_claude_stdout.txt" 2>&1 || true
    fi

    "$PYTHON_BIN" "$(dirname "$0")/redact.py" "$TASK_RAW_DIR" --in-place
    n_done=$((n_done + 1))
done < "$TASKS_FILE"

echo "[capture-skill] done: ${n_done}/${n_total} tasks captured under $RAW_ROOT"
