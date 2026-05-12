"""Tests for scripts/download_claudecode_trace.py."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

# scripts/ isn't a package, so import via spec.
_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "download_claudecode_trace.py"
_spec = importlib.util.spec_from_file_location("download_claudecode_trace", _SCRIPT)
_mod = importlib.util.module_from_spec(_spec)
sys.modules["download_claudecode_trace"] = _mod
_spec.loader.exec_module(_mod)


def test_validate_subsets_returns_empty_when_all_present(tmp_path):
    for s in ("swebench_verified", "skill_invocation", "post_compact"):
        (tmp_path / s).mkdir()
    assert _mod.validate_subsets(tmp_path) == set()


def test_validate_subsets_reports_missing(tmp_path):
    (tmp_path / "swebench_verified").mkdir()
    # skill_invocation, post_compact missing
    missing = _mod.validate_subsets(tmp_path)
    assert missing == {"skill_invocation", "post_compact"}


def test_validate_subsets_handles_nonexistent_dir(tmp_path):
    missing = _mod.validate_subsets(tmp_path / "does-not-exist")
    assert missing == set(_mod.EXPECTED_SUBSETS)


def test_stage_dataset_invokes_snapshot_download_with_repo_args(tmp_path):
    captured = {}

    def fake_snapshot(*, repo_id, repo_type, local_dir, revision, token):
        captured["repo_id"] = repo_id
        captured["repo_type"] = repo_type
        captured["local_dir"] = local_dir
        captured["revision"] = revision
        captured["token"] = token
        # Simulate snapshot_download writing into local_dir
        for s in ("swebench_verified", "skill_invocation", "post_compact"):
            (Path(local_dir) / s).mkdir(parents=True, exist_ok=True)
        return local_dir

    staged = _mod.stage_dataset(
        tmp_path / "stage", revision="main", token="hf_test_token",
        snapshot_download_fn=fake_snapshot,
    )
    assert captured["repo_id"] == "intelchen/claudecode-trace"
    assert captured["repo_type"] == "dataset"
    assert captured["revision"] == "main"
    assert captured["token"] == "hf_test_token"
    assert staged.exists()
    assert _mod.validate_subsets(staged) == set()


def test_stage_dataset_returns_resolved_path(tmp_path):
    def fake_snapshot(*, repo_id, repo_type, local_dir, revision, token):
        return local_dir
    staged = _mod.stage_dataset(
        tmp_path / "stage", snapshot_download_fn=fake_snapshot,
    )
    assert isinstance(staged, Path)
