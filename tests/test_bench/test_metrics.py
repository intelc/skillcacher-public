"""Tests for aggregate decomposed metrics across many records."""
from skillcacher.bench.metrics import aggregate, AggregateRates
from skillcacher.bench.replay import DecomposedRates


def _rate(skill_h, skill_t, ns_h, ns_t):
    return DecomposedRates(
        skill_hit_rate=skill_h / max(skill_t, 1),
        non_skill_hit_rate=ns_h / max(ns_t, 1),
        overall_hit_rate=(skill_h + ns_h) / max(skill_t + ns_t, 1),
        skill_hit_tokens=skill_h,
        non_skill_hit_tokens=ns_h,
        total_hit_tokens=skill_h + ns_h,
        skill_total_tokens=skill_t,
        non_skill_total_tokens=ns_t,
        prompt_token_count=skill_t + ns_t,
    )


def test_aggregate_sums_token_pools_then_divides():
    rates = [
        _rate(skill_h=30, skill_t=60, ns_h=10, ns_t=40),  # 40/100 overall
        _rate(skill_h=50, skill_t=80, ns_h=20, ns_t=20),  # 70/100 overall
    ]
    agg = aggregate(rates)
    assert isinstance(agg, AggregateRates)
    assert agg.skill_hit_rate == 80 / 140  # (30+50)/(60+80)
    assert agg.non_skill_hit_rate == 30 / 60  # (10+20)/(40+20)
    assert agg.overall_hit_rate == (40 + 70) / 200  # (30+10+50+20)/(60+40+80+20)
    assert agg.n_requests == 2


def test_aggregate_handles_empty():
    agg = aggregate([])
    assert agg.skill_hit_rate == 0.0
    assert agg.n_requests == 0
