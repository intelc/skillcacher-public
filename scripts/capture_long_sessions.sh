#!/usr/bin/env bash
# the harness Layer 1: capture real Claude Code traffic on SWE-Bench Verified
# long tasks (>150K projected context).
#
# Per-task flow:
# 1. Pull the task's problem_statement + hints_text from the cached SWE-V
#    parquet (scripts/dev/rank_swebench_verified.py runs once to populate it).
# 2. Spawn `claude -p` non-interactively against ANTHROPIC_BASE_URL=<proxy>
#    with the problem text as the prompt and a --max-budget-usd cap so a
#    runaway task can't burn the API balance.
# 3. SKILLCACHER_TRACE_DIR set so the proxy's trace_store writes per-request
#    parquets into the per-task raw directory.
# 4. After the run, grep for compaction markers and redact in place before
#    moving into the public fixtures tree.
#
# Limitations vs the project spec's E2 goal:
# - `claude -p` is one-shot non-interactive; CC runs its full agent loop
#   internally and exits when done. This produces real agentic traffic with
#   real tool calls, but won't necessarily span >200K tokens (the natural-
#   compaction threshold). When natural compaction doesn't fire, fall back
#   to scripts/synthetic_compaction.py to inject the trigger between captures.
# - We don't clone the SWE-V repo to a fresh tmp dir — CC works against the
#   user's CWD. Treat fixtures as "CC explores PWD against the SWE-V task
#   description" rather than "CC fixes SWE-V end-to-end."
#
# Requires: a `prefix_cache` pod up; ANTHROPIC_BASE_URL pointed at the proxy.
set -euo pipefail

ENV_FILE="${ENV_FILE:-$(dirname "$0")/../.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi

: "${ANTHROPIC_BASE_URL:?must point at the local skillcacher proxy}"

TASKS_FILE="${TASKS_FILE:-scripts/swebench_verified_long_tasks.txt}"
OUT_ROOT="${OUT_ROOT:-tests/fixtures/claude_code_real/swebench_verified}"
RAW_ROOT="${RAW_ROOT:-tests/fixtures/claude_code_real/_raw}"
TIMEOUT_SECS="${TIMEOUT_SECS:-3600}"
MAX_BUDGET_USD="${MAX_BUDGET_USD:-2}"
SWEV_PARQUET="${SWEV_PARQUET:-/tmp/swebench_verified.parquet}"

# Portability: macOS doesn't ship GNU coreutils, so `timeout` is missing. Use
# `gtimeout` if installed (brew install coreutils), else run without a wall-
# clock wrapper and rely on --max-budget-usd as the soft cap.
if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="gtimeout"
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout"
else
    TIMEOUT_BIN=""
    echo "[capture-long] note: no timeout binary; relying on --max-budget-usd as soft cap" >&2
fi
# Python: macOS's default Python is `python3`, not `python`. Prefer the repo's
# venv if present.
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# Compaction marker — CC's autocompact emits a recognizable summary block
# at the start of post-compact system prompts. Adjust the pattern from the
# first capture's empirical shape if grep misses real compactions.
COMPACT_MARKER="${COMPACT_MARKER:-Conversation summary}"

if [[ ! -f "$TASKS_FILE" ]]; then
    echo "[capture-long] tasks file $TASKS_FILE not found" >&2
    echo "[capture-long] populate it with one SWE-Bench Verified task ID per line" >&2
    exit 2
fi

if [[ ! -f "$SWEV_PARQUET" ]]; then
    echo "[capture-long] SWE-V parquet not cached at $SWEV_PARQUET" >&2
    echo "[capture-long] run: python -m scripts.dev.rank_swebench_verified --top 1" >&2
    echo "[capture-long] (any invocation downloads the parquet as a side-effect)" >&2
    exit 2
fi

mkdir -p "$OUT_ROOT" "$RAW_ROOT"

# Helper: extract problem_statement + hints_text for one task ID from the
# parquet. Echoes a multi-line prompt to stdout.
build_prompt_for_task() {
    local task_id="$1"
    "$PYTHON_BIN" - <<PY
import sys
import pyarrow.parquet as pq
table = pq.read_table("$SWEV_PARQUET")
cols = table.column_names
for i in range(table.num_rows):
    row = {k: table[k][i].as_py() for k in cols}
    if row["instance_id"] != "$task_id":
        continue
    repo = row.get("repo", "")
    ps = (row.get("problem_statement") or "").strip()
    hints = (row.get("hints_text") or "").strip()
    print(f"You are exploring a Python codebase to plan a fix for an upstream issue.")
    print(f"Source: SWE-Bench Verified task {row['instance_id']} (repo: {repo}).")
    print()
    print("# Problem statement")
    print()
    print(ps)
    if hints:
        print()
        print("# Hints from the issue thread")
        print()
        print(hints)
    print()
    print("Use whatever tools are available (Read, Glob, Grep, Bash) to explore"
          " the local code and propose a draft patch — do not actually apply or"
          " commit changes; produce a written plan describing the fix.")
    sys.exit(0)
print("ERROR: task not found", file=sys.stderr)
sys.exit(1)
PY
}

n_total=0
n_compacted=0
n_no_compaction=0

while IFS= read -r TASK; do
    [[ -z "$TASK" || "$TASK" =~ ^# ]] && continue
    n_total=$((n_total + 1))
    echo "[capture-long] $TASK (timeout=${TIMEOUT_SECS}s, budget=\$${MAX_BUDGET_USD})"

    TASK_RAW_DIR="$RAW_ROOT/$TASK"
    mkdir -p "$TASK_RAW_DIR"

    PROMPT_FILE="$(mktemp -t swev_prompt_XXXXXX)"
    if ! build_prompt_for_task "$TASK" > "$PROMPT_FILE"; then
        echo "[capture-long] $TASK: failed to build prompt; skipping"
        rm -f "$PROMPT_FILE"
        continue
    fi

    # Run CC non-interactively under --bare to drop CC's machinery overhead
    # (CLAUDE.md auto-discovery, plugins, hooks, auto-memory, keychain reads).
    # the harness finding: real CC -p without --bare sends ~50K-token prompts;
    # --bare cuts that ~80%. ANTHROPIC_API_KEY must be set explicitly under
    # --bare (OAuth + keychain are skipped); our proxy doesn't auth-check, so
    # any non-empty value passes through.
    export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-EMPTY}"
    if [[ -n "$TIMEOUT_BIN" ]]; then
        SKILLCACHER_TRACE_DIR="$TASK_RAW_DIR" "$TIMEOUT_BIN" "$TIMEOUT_SECS" \
            claude --bare -p "$(cat "$PROMPT_FILE")" \
                --max-budget-usd "$MAX_BUDGET_USD" \
                --dangerously-skip-permissions \
                > "$TASK_RAW_DIR/_claude_stdout.txt" 2>&1 || true
    else
        SKILLCACHER_TRACE_DIR="$TASK_RAW_DIR" \
            claude --bare -p "$(cat "$PROMPT_FILE")" \
                --max-budget-usd "$MAX_BUDGET_USD" \
                --dangerously-skip-permissions \
                > "$TASK_RAW_DIR/_claude_stdout.txt" 2>&1 || true
    fi
    rm -f "$PROMPT_FILE"

    if grep -rq "$COMPACT_MARKER" "$TASK_RAW_DIR" 2>/dev/null; then
        echo "[capture-long] $TASK: ✓ natural compaction detected"
        n_compacted=$((n_compacted + 1))
    else
        echo "[capture-long] $TASK: ✗ no natural compaction in trace"
        echo "[capture-long]   fallback: python -m scripts.synthetic_compaction --out /tmp/stub.txt"
        n_no_compaction=$((n_no_compaction + 1))
    fi

    "$PYTHON_BIN" "$(dirname "$0")/redact.py" "$TASK_RAW_DIR" --in-place
    # the harness followup skip the staging→promote `mv` when the
    # caller (typically capture_orchestrator.py) already pointed
    # RAW_ROOT at the final fixture dir. Without this guard, mv would
    # error with "Invalid argument" on macOS for a same-path rename
    # AND `_dump_logs` (which runs in the orchestrator's __aexit__,
    # AFTER this loop returns) would land vllm.log + oneshot_boot.log
    # inside a now-empty TASK_RAW_DIR that just got moved away.
    if [[ "$(cd "$TASK_RAW_DIR" && pwd -P)" != "$(cd "$(dirname "$OUT_ROOT/$TASK")" && pwd -P)/$(basename "$OUT_ROOT/$TASK")" ]]; then
        mkdir -p "$(dirname "$OUT_ROOT/$TASK")"
        rm -rf "$OUT_ROOT/$TASK"
        mv "$TASK_RAW_DIR" "$OUT_ROOT/$TASK"
    else
        echo "[capture-long] $TASK: TASK_RAW_DIR already at OUT_ROOT/$TASK; skipping mv"
    fi
done < "$TASKS_FILE"

echo "[capture-long] done. ${n_total} tasks: ${n_compacted} compacted naturally, ${n_no_compaction} need synthetic trigger"
echo "[capture-long] captures redacted into $OUT_ROOT/"
