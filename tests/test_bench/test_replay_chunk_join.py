"""Tests for chunk-level join replay (no proportional attribution)."""
from pathlib import Path

from skillcacher.bench.replay import compute_decomposed_hit_rate
from skillcacher.proxy.trace_store import RequestRecord, ChunkHitMap


def _rec(prompt_tokens, span_tags, span_lookups, engine_total_hit=None):
    return RequestRecord(
        request_id="r1", session_id="s1",
        ts_start=0.0, ts_end=1.0,
        prompt_tokens=prompt_tokens, response_tokens=[],
        prompt_token_count=len(prompt_tokens), response_token_count=0,
        cache_read_tokens=engine_total_hit or 0, cache_recompute_tokens=0,
        ttft_ms=100.0, request_body_json="{}",
        span_tags=span_tags, span_lookups=span_lookups,
        engine_total_hit_tokens=engine_total_hit,
    )


def test_skill_hit_rate_from_chunk_lookups():
    # 100 prompt tokens, 60 are skill_body, 40 are system_static
    rec = _rec(
        prompt_tokens=list(range(100)),
        span_tags=["skill_body"] * 60 + ["system_static"] * 40,
        span_lookups=[
            ChunkHitMap("skill:a:1024", chunk_hits=2, total_chunks=4, hit_tokens=30, total_tokens=60),
        ],
        engine_total_hit=50,
    )
    rates = compute_decomposed_hit_rate(rec, span_id_to_tag={"skill:a:1024": "skill_body"})
    assert rates.skill_hit_rate == 0.5  # 30/60
    assert rates.non_skill_hit_rate == 0.5  # (50-30) / (100-60) = 20/40
    assert rates.overall_hit_rate == 0.5  # 50/100


def test_no_engine_total_falls_back_to_lookup_sum():
    rec = _rec(
        prompt_tokens=list(range(100)),
        span_tags=["skill_body"] * 60 + ["other"] * 40,
        span_lookups=[
            ChunkHitMap("skill:a:1024", chunk_hits=2, total_chunks=4, hit_tokens=30, total_tokens=60),
        ],
        engine_total_hit=None,  # stdout-tail timed out
    )
    rates = compute_decomposed_hit_rate(rec, span_id_to_tag={"skill:a:1024": "skill_body"})
    assert rates.skill_hit_rate == 0.5
    # When engine_total is None, total_hit_tokens = sum(lookup hits) = 30, so non_skill = 0
    assert rates.non_skill_hit_rate == 0.0


def test_zero_skill_tokens_returns_zero_rate_not_error():
    rec = _rec(
        prompt_tokens=list(range(40)),
        span_tags=["other"] * 40,
        span_lookups=[],
        engine_total_hit=0,
    )
    rates = compute_decomposed_hit_rate(rec, span_id_to_tag={})
    assert rates.skill_hit_rate == 0.0
    assert rates.overall_hit_rate == 0.0
