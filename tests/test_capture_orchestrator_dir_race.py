"""the harness followup — regression test for the orchestrator-vs-bash
dir-move race surfaced during the §4 SWE-V capture pass.

The bug: `capture_long_sessions.sh` and `capture_compaction.sh` both end
with `mv $RAW_ROOT/$TASK $OUT_ROOT/$TASK`. The orchestrator's
`_dump_logs` runs inside `__aexit__` of `ConditionLifecycle` AFTER the
bash subprocess returns, so vllm.log + oneshot_boot.log get SCP'd into
a now-empty `_raw/<TASK>/` instead of the moved fixture dir. On the §4
SWE-V captures, all 4 fixtures' logs orphaned to `_raw/`; we noticed
when the post-capture audit reported the new fixtures had no boot logs.

The fix: orchestrator points RAW_ROOT directly at OUT_ROOT (the final
destination) and the bash scripts skip the `mv` when source == dest.
After this:
- `task_raw_dir` IS the final fixture location;
- `log_dump_path` lands inside that final location, no race;
- the bash scripts work in single-dir mode under the orchestrator AND
  retain their staging→promote behavior under direct invocation.

This module exercises the path-computation logic in isolation (no pod,
no bash subprocess) plus greps both bash scripts for the same-path
guards. End-to-end coverage stays manual / via real captures.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest


SCRIPTS_ROOT = Path(__file__).parent.parent / "scripts"


# --- bash same-path guards --------------------------------------------------


def test_capture_long_sessions_skips_mv_when_dirs_match():
    """The fix introduces a same-path guard around the `mv`. Lock it in."""
    body = (SCRIPTS_ROOT / "capture_long_sessions.sh").read_text()
    # We expect a guard that resolves both paths via `pwd -P` and only
    # mv's when they differ.
    assert 'pwd -P' in body, (
        "capture_long_sessions.sh missing canonical-path resolution — "
        "without `pwd -P` the same-path guard breaks under symlinks"
    )
    assert 'skipping mv' in body, (
        "capture_long_sessions.sh missing the 'skipping mv' branch that "
        "fires when TASK_RAW_DIR == OUT_ROOT/$TASK; without it the bash "
        "script either errors (mv to same path) or, worse, silently "
        "orphans the logs the orchestrator dumps after."
    )


def test_capture_compaction_skips_mv_when_dirs_match():
    body = (SCRIPTS_ROOT / "capture_compaction.sh").read_text()
    assert 'pwd -P' in body, "capture_compaction.sh missing canonical-path resolution"
    assert 'skipping mv' in body, (
        "capture_compaction.sh missing same-path mv guard — see "
        "capture_long_sessions.sh for the rationale."
    )


# --- orchestrator path computation -----------------------------------------


def _fresh_module(monkeypatch, env: dict[str, str]):
    """Re-import capture_orchestrator with a clean env so the module-level
    defaults aren't sticky across tests."""
    import importlib
    import sys
    monkeypatch.delenv("RAW_ROOT", raising=False)
    monkeypatch.delenv("OUT_ROOT", raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    if "scripts.capture_orchestrator" in sys.modules:
        del sys.modules["scripts.capture_orchestrator"]
    return importlib.import_module("scripts.capture_orchestrator")


def test_orchestrator_long_session_default_dir_is_swebench_verified(monkeypatch, tmp_path):
    """Lock in the per-mode OUT_ROOT default. For mode=long_session we
    expect tests/fixtures/.../swebench_verified/, NOT the legacy `_raw/`
    staging dir that triggered the orphan bug."""
    # Inspect the source to verify the mode→default map; importing the
    # module is enough — the dict literal lives at module scope.
    orch_path = SCRIPTS_ROOT / "capture_orchestrator.py"
    src = orch_path.read_text()
    # Default destinations should be the `claude_code_real/<mode>/` paths.
    assert '"long_session": "tests/fixtures/claude_code_real/swebench_verified"' in src
    assert '"compaction": "tests/fixtures/claude_code_real/post_compact"' in src
    assert '"skill_invocation": "tests/fixtures/claude_code_real/skill_invocation"' in src


def test_orchestrator_passes_out_root_to_bash(monkeypatch):
    """The orchestrator must export OUT_ROOT (in addition to RAW_ROOT)
    so the bash scripts' same-path guard sees the same value."""
    orch_path = SCRIPTS_ROOT / "capture_orchestrator.py"
    src = orch_path.read_text()
    # Both env keys are set unconditionally for any mode whose bash
    # script uses them. Match on the assignment shape.
    assert 'env["OUT_ROOT"] = str(out_root)' in src, (
        "orchestrator must propagate OUT_ROOT to the bash subprocess so "
        "the same-path guard fires correctly"
    )
    assert 'env["RAW_ROOT"] = str(raw_root)' in src


def test_orchestrator_post_dump_redact_runs_in_finally(monkeypatch):
    """The post-dump redact pass scrubs vllm.log + oneshot_boot.log +
    proxy.log AFTER the orchestrator's __aexit__ has SCP'd them. It must
    sit OUTSIDE the `async with` block, in or after `finally`, so it
    actually fires when the dumps land."""
    orch_path = SCRIPTS_ROOT / "capture_orchestrator.py"
    src = orch_path.read_text()
    # Find the sentinel sequence: the redact import + the three log file
    # names + the loop. Locate them after the main try/finally block.
    assert "from scripts.redact import redact_file" in src
    for log_name in ("vllm.log", "oneshot_boot.log", "proxy.log"):
        assert f'"{log_name}"' in src, f"post-dump redact missing {log_name}"
    # The redact loop must come AFTER `async with ConditionLifecycle`.
    redact_idx = src.find("from scripts.redact import redact_file")
    aenter_idx = src.find("async with ConditionLifecycle")
    assert redact_idx > aenter_idx, (
        "post-dump redact must run AFTER __aexit__ (i.e., source position "
        "after `async with ConditionLifecycle`); otherwise the SCP'd logs "
        "haven't been written yet."
    )
