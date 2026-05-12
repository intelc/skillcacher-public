"""Structural tests for scripts/capture_compaction.sh — the harness
multi-turn /compact capture flow.

Running the script end-to-end requires a live `claude` binary and a
backend pod, so these tests verify the *shape* of the script: it parses,
it uses --resume + --session-id correctly, it issues a /compact turn, it
captures a post-compact continuation, and it sets SKILLCACHER_TRACE_DIR
on every claude invocation (the harness spec correction #6 — must plant
the trace dir before the proxy reads it)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "capture_compaction.sh"


def _read() -> str:
    return SCRIPT_PATH.read_text()


def test_script_is_executable():
    assert SCRIPT_PATH.exists(), f"missing {SCRIPT_PATH}"
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & 0o111, "capture_compaction.sh must be executable"


def test_script_bash_parses():
    if not shutil.which("bash"):
        pytest.skip("bash not on PATH")
    r = subprocess.run(
        ["bash", "-n", str(SCRIPT_PATH)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, f"bash -n failed: {r.stderr}"


def test_script_uses_session_id_on_first_turn():
    """Turn 1 must use --session-id (not --resume) so the session is
    actually created."""
    src = _read()
    assert "--session-id" in src
    assert "$SESSION_ID" in src or '"$SESSION_ID"' in src


def test_script_uses_resume_on_subsequent_turns():
    """Turns 2+ must use --resume against the same session id, otherwise
    the conversation history doesn't accumulate."""
    src = _read()
    assert "--resume" in src


def test_script_issues_compact_turn():
    """The whole point of the spike: there must be a /compact invocation
    against the resumed session."""
    src = _read()
    assert "/compact" in src


def test_script_captures_post_compact_continuation():
    """The post-compact turn is the headline measurement target — its
    request body is what the bench harness will replay across conditions."""
    src = _read()
    # Either the default POST_COMPACT_PROMPT var, or a marker like
    # `_postcompact` for the per-turn stdout file.
    assert "POST_COMPACT_PROMPT" in src or "_postcompact" in src


def test_script_plants_trace_dir_per_turn():
    """the harness spec correction #6: SKILLCACHER_TRACE_DIR must be set
    on every `claude` invocation so per-request token parquets land in the
    per-task fixture dir, not the bench's default location."""
    src = _read()
    # The helper run_turn function should reference the env var, and the
    # var name must appear at least once in a position that affects claude.
    assert "SKILLCACHER_TRACE_DIR" in src
    # At least one occurrence should be in a position that prefixes a
    # claude invocation — the run_turn helper does this once.
    assert 'SKILLCACHER_TRACE_DIR="$TASK_RAW_DIR"' in src


def test_script_writes_meta_json_with_provenance():
    """meta.json must declare compaction_source so downstream consumers
    (bench harness, dataset publication) can tag this datapoint correctly."""
    src = _read()
    assert "compaction_source" in src
    assert "real_cc_compact" in src


def test_script_runs_redact_in_place_before_promoting():
    """Captures land under _raw/ and must pass redact.py before being
    moved to the public fixture tree, mirroring capture_long_sessions.sh."""
    src = _read()
    assert "redact.py" in src
    # Promote pattern: rm -rf $OUT/$TASK then mv $RAW/$TASK $OUT/$TASK.
    assert "rm -rf" in src
    assert "mv " in src


def test_script_exits_on_too_few_prompts():
    """Defensive: at least 2 build-up turns are required so /compact has
    something substantive to summarize. Running with one turn is a config
    error and should fail loudly."""
    if not shutil.which("bash"):
        pytest.skip("bash not on PATH")
    env = os.environ.copy()
    env["ANTHROPIC_BASE_URL"] = "http://127.0.0.1:9999"  # not actually called
    # Pass a single-prompt file via TURN_PROMPTS_FILE; should exit with rc=2
    # before any claude invocation, never reaching the network.
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tf:
        tf.write("only one prompt\n")
        tf_path = tf.name
    try:
        env["TURN_PROMPTS_FILE"] = tf_path
        env["RAW_ROOT"] = "/tmp/skillcacher_test_raw"
        env["OUT_ROOT"] = "/tmp/skillcacher_test_out"
        # Block real `claude`/`uuidgen` calls by giving an empty PATH —
        # the script must exit on too-few-prompts BEFORE it hits any
        # external binary.
        env["PATH"] = "/usr/bin:/bin"  # keep cat/uuidgen for header
        r = subprocess.run(
            ["bash", str(SCRIPT_PATH)],
            env=env, capture_output=True, text=True, timeout=10,
        )
        assert r.returncode == 2, (
            f"expected rc=2 on too-few-prompts; got {r.returncode}\n"
            f"stdout={r.stdout}\nstderr={r.stderr}"
        )
        assert "need at least 2 build-up turns" in r.stderr
    finally:
        Path(tf_path).unlink(missing_ok=True)
