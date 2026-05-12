"""integration test for statistical-source span registry path.

Asserts source="statistical" entries flow through register → get →
all_spans identically to "structural" entries, that lookup is
source-agnostic (no filtering by source), and that the §3 mining →
register pipeline lands rows we can read back.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from skillcacher.registry.span_registry import SpanRegistry
from skillcacher.tagging.statistical_miner import MinedSpan, mine_spans


@pytest.fixture
def registry(tmp_path: Path) -> SpanRegistry:
    r = SpanRegistry(tmp_path / "spans.sqlite")
    r.init_schema()
    return r


def test_register_with_source_statistical(registry):
    registry.register(
        "stat:abc123", token_ids=[5, 6, 7],
        skill_id="_statistical", anchor_bytes=0,
        source="statistical",
    )
    s = registry.get("stat:abc123")
    assert s is not None
    assert s.source == "statistical"
    assert s.token_ids == [5, 6, 7]


def test_register_default_source_is_structural(registry):
    registry.register("skill:foo:1024", token_ids=[1, 2, 3],
                      skill_id="foo", anchor_bytes=1024)
    s = registry.get("skill:foo:1024")
    assert s.source == "structural"


def test_register_rejects_unknown_source(registry):
    with pytest.raises(ValueError, match="source must be"):
        registry.register("x", token_ids=[1], skill_id="x",
                          anchor_bytes=0, source="invented")


def test_all_spans_filters_by_source(registry):
    registry.register("a", token_ids=[1], skill_id="foo", anchor_bytes=1024)
    registry.register("b", token_ids=[2], skill_id="bar", anchor_bytes=2048)
    registry.register("c", token_ids=[3], skill_id="_statistical",
                      anchor_bytes=0, source="statistical")
    registry.register("d", token_ids=[4], skill_id="_statistical",
                      anchor_bytes=0, source="statistical")

    structural = registry.all_spans(source="structural")
    statistical = registry.all_spans(source="statistical")
    everything = registry.all_spans()

    assert {s.span_id for s in structural} == {"a", "b"}
    assert {s.span_id for s in statistical} == {"c", "d"}
    assert {s.span_id for s in everything} == {"a", "b", "c", "d"}


def test_spans_referenced_by_is_source_agnostic(registry):
    """The spec says lookup is source-agnostic; verify
    spans_referenced_by(skill_ids=...) returns matching rows regardless
    of their source flag."""
    registry.register("a", token_ids=[1], skill_id="shared", anchor_bytes=1024)
    registry.register("b", token_ids=[2], skill_id="shared", anchor_bytes=2048,
                      source="statistical")
    rows = registry.spans_referenced_by(skill_ids=["shared"])
    assert {s.span_id for s in rows} == {"a", "b"}
    sources = {s.span_id: s.source for s in rows}
    assert sources == {"a": "structural", "b": "statistical"}


def test_record_lookup_works_for_statistical(registry):
    registry.register(
        "stat:xyz", token_ids=[10, 20, 30],
        skill_id="_statistical", anchor_bytes=0,
        source="statistical",
    )
    registry.record_lookup_result("stat:xyz", hit_tokens=3, total_tokens=3)
    s = registry.get("stat:xyz")
    assert s.lookup_hit_count == 3
    assert s.lookup_total_count == 1
    assert s.source == "statistical"  # unchanged


def test_miner_to_registry_pipeline(registry):
    """End-to-end: mine planted repeats from synthetic streams, register
    each MinedSpan with source='statistical', read back and confirm the
    rows match."""
    import random
    rng = random.Random(7)
    planted = [rng.randrange(0, 50_000) for _ in range(300)]
    streams = [
        [rng.randrange(0, 50_000) for _ in range(200)] + planted +
        [rng.randrange(0, 50_000) for _ in range(200)],
        planted + [rng.randrange(0, 50_000) for _ in range(500)],
        [rng.randrange(0, 50_000) for _ in range(100)] + planted +
        [rng.randrange(0, 50_000) for _ in range(100)],
    ]
    spans = mine_spans(streams, length_floor=256, frequency_floor=3)
    assert len(spans) >= 1

    for ms in spans:
        registry.register(
            f"stat:{ms.fingerprint()}",
            token_ids=list(ms.token_ids),
            skill_id="_statistical",
            anchor_bytes=0,
            source="statistical",
        )

    stat_rows = registry.all_spans(source="statistical")
    assert len(stat_rows) == len(spans)
    # Token sequences round-trip identically.
    by_id = {s.span_id: s for s in stat_rows}
    for ms in spans:
        row = by_id[f"stat:{ms.fingerprint()}"]
        assert tuple(row.token_ids) == ms.token_ids
        assert row.source == "statistical"


def test_schema_migration_adds_source_to_existing_db(tmp_path: Path):
    """the harness added the `source` column. A registry created before
    that change has no `source` column; init_schema() must add it via
    ALTER TABLE rather than failing."""
    db_path = tmp_path / "old.sqlite"
    # Hand-build the pre-§3 schema.
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE spans (
            span_id TEXT PRIMARY KEY,
            skill_id TEXT NOT NULL,
            anchor_bytes INTEGER NOT NULL,
            token_ids BLOB NOT NULL,
            registered_at REAL NOT NULL,
            last_lookup_at REAL,
            lookup_hit_count INTEGER NOT NULL DEFAULT 0,
            lookup_total_count INTEGER NOT NULL DEFAULT 0
        );
        INSERT INTO spans
          (span_id, skill_id, anchor_bytes, token_ids, registered_at)
          VALUES ('legacy:1', 'old-skill', 1024, X'00000001', 0.0);
        """)

    # init_schema() should add the source column without losing data.
    r = SpanRegistry(db_path)
    r.init_schema()

    s = r.get("legacy:1")
    assert s is not None
    # Default value for the migrated column is 'structural'.
    assert s.source == "structural"
    # Subsequent register calls work.
    r.register("new:1", token_ids=[2, 3], skill_id="_statistical",
               anchor_bytes=0, source="statistical")
    s2 = r.get("new:1")
    assert s2.source == "statistical"
