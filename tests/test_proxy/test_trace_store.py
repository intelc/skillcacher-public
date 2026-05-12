import json
from pathlib import Path

import pytest

from skillcacher.proxy.trace_store import TraceStore, RequestRecord, ChunkHitMap


def test_trace_store_roundtrip(tmp_path):
    store = TraceStore(tmp_path)
    store.init_schema()

    rec = RequestRecord(
        request_id="r1",
        session_id="s1",
        ts_start=1.0,
        ts_end=1.5,
        prompt_tokens=[1, 2, 3, 4, 5],
        response_tokens=[6, 7, 8],
        prompt_token_count=5,
        response_token_count=3,
        cache_read_tokens=2,
        cache_recompute_tokens=3,
        ttft_ms=120.0,
        request_body_json=json.dumps({"model": "test"}),
    )
    store.write(rec)

    rows = list(store.read_all())
    assert len(rows) == 1
    assert rows[0].request_id == "r1"
    assert rows[0].prompt_tokens == [1, 2, 3, 4, 5]


def test_trace_store_no_double_init(tmp_path):
    store = TraceStore(tmp_path)
    store.init_schema()
    store.init_schema()  # idempotent


def test_trace_store_span_tags_roundtrip(tmp_path):
    store = TraceStore(tmp_path)
    store.init_schema()

    rec = RequestRecord(
        request_id="r2",
        session_id="s2",
        ts_start=2.0,
        ts_end=2.5,
        prompt_tokens=[10, 20, 30, 40],
        response_tokens=[50, 60],
        prompt_token_count=4,
        response_token_count=2,
        cache_read_tokens=1,
        cache_recompute_tokens=1,
        ttft_ms=80.0,
        request_body_json=json.dumps({"model": "test"}),
        span_tags=["skill_body", "skill_body", "tool_def", "other"],
    )
    store.write(rec)

    rows = list(store.read_all())
    assert len(rows) == 1
    assert rows[0].prompt_tokens == [10, 20, 30, 40]
    assert rows[0].span_tags == ["skill_body", "skill_body", "tool_def", "other"]


def _make_store(tmp_path: Path) -> TraceStore:
    store = TraceStore(tmp_path)
    store.init_schema()
    return store


def _base_record(**overrides) -> RequestRecord:
    rec = RequestRecord(
        request_id="req_test1", session_id="sess_test1",
        ts_start=0.0, ts_end=0.1,
        prompt_tokens=[1, 2, 3], response_tokens=[],
        prompt_token_count=3, response_token_count=0,
        cache_read_tokens=0, cache_recompute_tokens=0,
        ttft_ms=100.0,
        request_body_json="{}",
        span_tags=["other", "other", "other"],
    )
    for k, v in overrides.items():
        setattr(rec, k, v)
    return rec


def test_write_accepts_valid_record(tmp_path):
    store = _make_store(tmp_path)
    store.write(_base_record())
    rows = list(store.read_all())
    assert len(rows) == 1


def test_write_rejects_mismatched_tag_length(tmp_path):
    store = _make_store(tmp_path)
    bad = _base_record(span_tags=["other"])  # only 1 tag for 3 tokens
    with pytest.raises(ValueError, match="span_tags length"):
        store.write(bad)


def test_write_rejects_negative_token_id(tmp_path):
    store = _make_store(tmp_path)
    bad = _base_record(prompt_tokens=[1, -1, 3], span_tags=["other", "other", "other"])
    with pytest.raises(ValueError, match="negative token id"):
        store.write(bad)


def test_chunk_hit_map_dataclass_round_trips(tmp_path):
    store = _make_store(tmp_path)
    chm = ChunkHitMap(span_id="skill:foo:1024", chunk_hits=2, total_chunks=4, hit_tokens=512, total_tokens=1024)
    rec = _base_record(span_lookups=[chm])
    store.write(rec)
    out = next(store.read_all())
    assert len(out.span_lookups) == 1
    assert out.span_lookups[0].span_id == "skill:foo:1024"
    assert out.span_lookups[0].hit_tokens == 512


def test_write_warns_on_lookup_sum_exceeds_engine_total(tmp_path, caplog):
    store = _make_store(tmp_path)
    chm1 = ChunkHitMap(span_id="a", chunk_hits=1, total_chunks=1, hit_tokens=200, total_tokens=200)
    chm2 = ChunkHitMap(span_id="b", chunk_hits=1, total_chunks=1, hit_tokens=300, total_tokens=300)
    rec = _base_record(span_lookups=[chm1, chm2], engine_total_hit_tokens=400)
    store.write(rec)  # does not raise
    out = next(store.read_all())
    assert "lookup_sum_gt_engine_total" in out.invariant_violations


def test_write_then_read_preserves_response_tokens_and_cache_counts(tmp_path):
    store = _make_store(tmp_path)
    rec = _base_record(
        response_tokens=[10, 11, 12],
        cache_read_tokens=42,
        cache_recompute_tokens=7,
    )
    store.write(rec)
    out = next(store.read_all())
    assert out.response_tokens == [10, 11, 12]
    assert out.cache_read_tokens == 42
    assert out.cache_recompute_tokens == 7


def test_chunk_hit_map_round_trips_all_fields(tmp_path):
    store = _make_store(tmp_path)
    chm = ChunkHitMap(span_id="skill:foo:1024", chunk_hits=2, total_chunks=4, hit_tokens=512, total_tokens=1024)
    rec = _base_record(span_lookups=[chm])
    store.write(rec)
    out = next(store.read_all())
    assert len(out.span_lookups) == 1
    got = out.span_lookups[0]
    assert got.span_id == "skill:foo:1024"
    assert got.chunk_hits == 2
    assert got.total_chunks == 4
    assert got.hit_tokens == 512
    assert got.total_tokens == 1024


def test_record_with_empty_tags_round_trips_to_empty(tmp_path):
    """Regression: span_tags=[] on write, must be [] (not [""]*N) on read.
    Otherwise a re-write would falsely trigger Invariant 1."""
    store = _make_store(tmp_path)
    rec = _base_record(span_tags=[])
    store.write(rec)
    out = next(store.read_all())
    assert out.span_tags == []  # not ['', '', ''] !
    # Re-writing should also work without raising Invariant 1
    out.request_id = "req_test2"  # avoid PRIMARY KEY collision
    store.write(out)


def test_invariant_3_does_not_double_append_on_retry(tmp_path):
    """Regression: write() with engine_total mismatch must use copy-on-write
    on invariant_violations so retry doesn't accumulate duplicates."""
    store = _make_store(tmp_path)
    chm1 = ChunkHitMap(span_id="a", chunk_hits=1, total_chunks=1, hit_tokens=200, total_tokens=200)
    chm2 = ChunkHitMap(span_id="b", chunk_hits=1, total_chunks=1, hit_tokens=300, total_tokens=300)
    rec = _base_record(span_lookups=[chm1, chm2], engine_total_hit_tokens=400)
    store.write(rec)
    # Caller still holds rec; mutate request_id and write again
    rec.request_id = "req_test2"
    store.write(rec)
    # Both records should have exactly ONE violation entry, not 1 and 2
    rows = list(store.read_all())
    assert len(rows) == 2
    assert all(r.invariant_violations.count("lookup_sum_gt_engine_total") == 1 for r in rows)
