#!/usr/bin/env bash
# capture real Claude Code /compact traffic via scripted multi-
# turn `claude --print --resume` sessions.
#
# The  spike confirmed /compact works in --print --continue mode
# (no pexpect/TTY needed). Each turn (including the /compact one) routes
# through ANTHROPIC_BASE_URL=our-proxy, so all of it lands in our trace
# store. Reference: docs/superpowers/specs/-skillcacher-plan4-design.md §1.
#
# Per-task flow:
# 1. Generate a fresh session id (`uuidgen`).
# 2. Turn 1 — initial prompt with `claude --bare -p --session-id $SID`.
# 3. Turns 2..N — continuation prompts with `claude --bare -p --resume $SID`.
# 4. Compaction — `/compact` as a `--resume` turn.
# 5. Post-compaction — one continuation `--resume` turn that captures the
#    post-compact request shape (the headline measurement target).
# 6. Per-turn stdout goes to `_turn_<n>_stdout.txt`; the proxy's trace_store
#    accumulates all per-request token parquets in $SKILLCACHER_TRACE_DIR.
# 7. Redact in place, then move into the public fixtures dir.
#
# Three branch outcomes for the Week 1 spike (see design spec §1 decision tree):
# - `/compact` succeeds and the post-compact request has a summary in the
#   `system` field → primary path validated; bench harness uses real captures.
# - `/compact` errors (Llama can't generate CC's expected output format)
#   → fall back to scripts/compaction_synth.py.
# - `/compact` succeeds but the post-compact request looks malformed
#   → investigate before committing.
#
# Requires: a pod up under any condition; ANTHROPIC_BASE_URL pointed at the
# local skillcacher proxy.
set -euo pipefail

ENV_FILE="${ENV_FILE:-$(dirname "$0")/../.env}"
if [[ -f "$ENV_FILE" ]]; then set -a; . "$ENV_FILE"; set +a; fi

: "${ANTHROPIC_BASE_URL:?must point at the local skillcacher proxy}"

TASK_ID="${TASK_ID:-compaction_spike_$(date +%Y%m%d_%H%M%S)}"
RAW_ROOT="${RAW_ROOT:-tests/fixtures/claude_code_real/_raw}"
OUT_ROOT="${OUT_ROOT:-tests/fixtures/claude_code_real/post_compact}"
TURN_BUDGET_USD="${TURN_BUDGET_USD:-0.5}"
COMPACT_BUDGET_USD="${COMPACT_BUDGET_USD:-1.0}"

# Default prompts build a multi-turn programming Q&A so /compact has
# substantive context to summarize. Override via TURN_PROMPTS_FILE (one
# prompt per line, blank lines skipped) for SWE-V-style content.
DEFAULT_PROMPTS=(
    "Explain what asyncio.gather does in Python and when you'd use it. Keep it under 4 paragraphs."
    "What is the practical difference between asyncio.gather and asyncio.as_completed? Give a small example for each."
    "Walk me through asyncio.TaskGroup from Python 3.11+. How does it differ from gather?"
    "What happens to sibling tasks if one task inside a TaskGroup raises an exception?"
    "How does asyncio.shield interact with cancellation? When is it useful?"
)
POST_COMPACT_PROMPT="${POST_COMPACT_PROMPT:-Summarize the asyncio concurrency primitives we discussed in two short bullet points each. No code blocks.}"

# Python: prefer the repo venv, else fall back.
if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    PYTHON_BIN="python"
fi

# Portable timeout: gtimeout (brew coreutils) > timeout > none.
if command -v gtimeout >/dev/null 2>&1; then
    TIMEOUT_BIN="gtimeout"
elif command -v timeout >/dev/null 2>&1; then
    TIMEOUT_BIN="timeout"
else
    TIMEOUT_BIN=""
fi
TURN_TIMEOUT_S="${TURN_TIMEOUT_S:-300}"

TASK_RAW_DIR="$RAW_ROOT/$TASK_ID"
mkdir -p "$TASK_RAW_DIR"

# Load prompts from file if given, else use the defaults.
PROMPTS=()
if [[ -n "${TURN_PROMPTS_FILE:-}" && -f "$TURN_PROMPTS_FILE" ]]; then
    while IFS= read -r line; do
        [[ -z "$line" || "$line" =~ ^# ]] && continue
        PROMPTS+=("$line")
    done < "$TURN_PROMPTS_FILE"
else
    PROMPTS=("${DEFAULT_PROMPTS[@]}")
fi

if (( ${#PROMPTS[@]} < 2 )); then
    echo "[capture-compact] need at least 2 build-up turns; got ${#PROMPTS[@]}" >&2
    exit 2
fi

SESSION_ID=$(uuidgen)
echo "$SESSION_ID" > "$TASK_RAW_DIR/_session_id.txt"

echo "[capture-compact] task=$TASK_ID session=$SESSION_ID raw_dir=$TASK_RAW_DIR" >&2
echo "[capture-compact] $((${#PROMPTS[@]})) build-up turns + /compact + 1 post-compact" >&2

# `claude --bare` skips OAuth + keychain; ANTHROPIC_API_KEY must be set
# explicitly. Our proxy doesn't auth-check, so any non-empty value passes.
export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-EMPTY}"

# Helper: run one claude turn. $1 = stdout file basename; remaining args =
# extra `claude` flags before the prompt; the prompt itself is read from
# stdin so it survives shell quoting.
run_turn() {
    local out_basename="$1"; shift
    local extra_flags=("$@")
    local prompt
    prompt=$(cat)
    local out_path="$TASK_RAW_DIR/${out_basename}_stdout.txt"
    local cmd=(claude --bare -p "$prompt" "${extra_flags[@]}"
               --max-budget-usd "$TURN_BUDGET_USD"
               --dangerously-skip-permissions)
    if [[ -n "$TIMEOUT_BIN" ]]; then
        SKILLCACHER_TRACE_DIR="$TASK_RAW_DIR" "$TIMEOUT_BIN" "$TURN_TIMEOUT_S" \
            "${cmd[@]}" > "$out_path" 2>&1 || true
    else
        SKILLCACHER_TRACE_DIR="$TASK_RAW_DIR" \
            "${cmd[@]}" > "$out_path" 2>&1 || true
    fi
}

# Turn 1 — initial, with --session-id.
echo "[capture-compact] turn 1 (session-id) ..." >&2
printf '%s' "${PROMPTS[0]}" | run_turn "_turn_1" --session-id "$SESSION_ID"

# Turns 2..N — --resume.
for i in $(seq 1 $(( ${#PROMPTS[@]} - 1 ))); do
    echo "[capture-compact] turn $((i+1)) (resume) ..." >&2
    printf '%s' "${PROMPTS[$i]}" | run_turn "_turn_$((i+1))" --resume "$SESSION_ID"
done

# Compaction — /compact as a resumed turn. Larger budget since /compact
# generates a multi-paragraph summary against the prior turns.
echo "[capture-compact] /compact turn ..." >&2
TURN_BUDGET_USD="$COMPACT_BUDGET_USD" \
    printf '/compact' | run_turn "_compact" --resume "$SESSION_ID"

# Post-compact — one continuation turn that captures the post-compact
# request shape (headline measurement target).
echo "[capture-compact] post-compact turn ..." >&2
printf '%s' "$POST_COMPACT_PROMPT" | run_turn "_postcompact" --resume "$SESSION_ID"

# Provenance file for downstream consumers (bench harness, dataset publish).
"$PYTHON_BIN" - "$TASK_RAW_DIR" "$SESSION_ID" "$TASK_ID" <<'PY'
import json
import sys
from pathlib import Path

raw_dir = Path(sys.argv[1])
session_id = sys.argv[2]
task_id = sys.argv[3]

(raw_dir / "meta.json").write_text(json.dumps({
    "task_id": task_id,
    "session_id": session_id,
    "compaction_source": "real_cc_compact",
    "schema_version": "plan4_postcompact_v1",
}, indent=2))
PY

# Redact + promote.
"$PYTHON_BIN" "$(dirname "$0")/redact.py" "$TASK_RAW_DIR" --in-place
# the harness followup skip the staging→promote `mv` when the
# caller (typically capture_orchestrator.py) already pointed RAW_ROOT
# at the final fixture dir. See capture_long_sessions.sh for the full
# rationale.
if [[ "$(cd "$TASK_RAW_DIR" && pwd -P)" != "$(cd "$OUT_ROOT" 2>/dev/null && pwd -P)/$TASK_ID" ]]; then
    mkdir -p "$OUT_ROOT"
    rm -rf "$OUT_ROOT/$TASK_ID"
    mv "$TASK_RAW_DIR" "$OUT_ROOT/$TASK_ID"
else
    echo "[capture-compact] TASK_RAW_DIR already at OUT_ROOT/$TASK_ID; skipping mv" >&2
fi

echo "[capture-compact] done — fixture at $OUT_ROOT/$TASK_ID" >&2
