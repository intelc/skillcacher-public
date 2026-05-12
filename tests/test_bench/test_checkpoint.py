"""Tests for the bench checkpoint manager — write/scan/resume from per-request JSON."""
from pathlib import Path

from skillcacher.bench.checkpoint import CheckpointStore


def test_write_and_scan_resumes_at_next_index(tmp_path):
    cs = CheckpointStore(tmp_path / "run1")
    cs.write_request("no_cache", 0, {"request_id": "r0"})
    cs.write_request("no_cache", 1, {"request_id": "r1"})
    cs.write_request("no_cache", 2, {"request_id": "r2"})
    next_idx = cs.next_index("no_cache")
    assert next_idx == 3


def test_next_index_zero_for_unstarted_condition(tmp_path):
    cs = CheckpointStore(tmp_path / "run1")
    assert cs.next_index("prefix_cache") == 0


def test_next_index_handles_gaps(tmp_path):
    cs = CheckpointStore(tmp_path / "run1")
    cs.write_request("cb", 0, {"r": 0})
    cs.write_request("cb", 2, {"r": 2})
    # We resume from the highest contiguous index + 1, not the max.
    # If 1 is missing, we re-do 1.
    assert cs.next_index("cb") == 1


def test_load_request_returns_dict(tmp_path):
    cs = CheckpointStore(tmp_path / "run1")
    cs.write_request("no_cache", 0, {"request_id": "r0", "hit_tokens": 100})
    out = cs.load_request("no_cache", 0)
    assert out["hit_tokens"] == 100


def test_iter_completed_yields_in_order(tmp_path):
    cs = CheckpointStore(tmp_path / "run1")
    for i in range(3):
        cs.write_request("c", i, {"i": i})
    out = list(cs.iter_completed("c"))
    assert [d["i"] for d in out] == [0, 1, 2]
