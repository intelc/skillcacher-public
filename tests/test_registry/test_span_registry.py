"""Tests for SpanRegistry — schema, register, lookup_history accounting, prune."""
import time
from pathlib import Path

import pytest

from skillcacher.registry.span_registry import SpanRegistry


@pytest.fixture
def registry(tmp_path: Path) -> SpanRegistry:
    r = SpanRegistry(tmp_path / "spans.sqlite")
    r.init_schema()
    return r


def test_register_and_get(registry):
    registry.register("skill:foo:1024", token_ids=[1, 2, 3], skill_id="foo", anchor_bytes=1024)
    s = registry.get("skill:foo:1024")
    assert s is not None
    assert s.token_ids == [1, 2, 3]
    assert s.skill_id == "foo"
    assert s.anchor_bytes == 1024


def test_register_replaces_existing(registry):
    registry.register("skill:foo:1024", token_ids=[1], skill_id="foo", anchor_bytes=1024)
    registry.register("skill:foo:1024", token_ids=[1, 2, 3], skill_id="foo", anchor_bytes=1024)
    s = registry.get("skill:foo:1024")
    assert s.token_ids == [1, 2, 3]


def test_record_lookup_result(registry):
    registry.register("skill:foo:1024", token_ids=[1, 2, 3], skill_id="foo", anchor_bytes=1024)
    registry.record_lookup_result("skill:foo:1024", hit_tokens=2, total_tokens=3)
    registry.record_lookup_result("skill:foo:1024", hit_tokens=3, total_tokens=3)
    s = registry.get("skill:foo:1024")
    assert s.lookup_total_count == 2
    assert s.lookup_hit_count == 5  # 2 + 3


def test_spans_referenced_by_returns_present_spans(registry):
    registry.register("skill:a:1024", token_ids=[1], skill_id="a", anchor_bytes=1024)
    registry.register("skill:b:1024", token_ids=[2], skill_id="b", anchor_bytes=1024)
    refs = registry.spans_referenced_by(skill_ids=["a"])
    ids = {s.span_id for s in refs}
    assert ids == {"skill:a:1024"}


def test_prune_removes_old_unused(registry):
    registry.register("skill:foo:1024", token_ids=[1], skill_id="foo", anchor_bytes=1024)
    # Force last_lookup_at to old timestamp
    import sqlite3
    with sqlite3.connect(registry.db_path) as conn:
        conn.execute("UPDATE spans SET last_lookup_at = ? WHERE span_id = ?", (1.0, "skill:foo:1024"))
    n = registry.prune(older_than=time.time() - 1)
    assert n == 1
    assert registry.get("skill:foo:1024") is None
