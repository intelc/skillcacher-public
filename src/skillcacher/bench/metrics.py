"""Aggregate decomposed hit-rate metrics across many records."""
from __future__ import annotations

from dataclasses import dataclass

from skillcacher.bench.replay import DecomposedRates


@dataclass
class AggregateRates:
    skill_hit_rate: float
    non_skill_hit_rate: float
    overall_hit_rate: float
    skill_hit_tokens: int
    non_skill_hit_tokens: int
    total_hit_tokens: int
    skill_total_tokens: int
    non_skill_total_tokens: int
    prompt_token_count: int
    n_requests: int


def aggregate(rates: list[DecomposedRates]) -> AggregateRates:
    if not rates:
        return AggregateRates(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    skill_h = sum(r.skill_hit_tokens for r in rates)
    skill_t = sum(r.skill_total_tokens for r in rates)
    ns_h = sum(r.non_skill_hit_tokens for r in rates)
    ns_t = sum(r.non_skill_total_tokens for r in rates)
    total_h = sum(r.total_hit_tokens for r in rates)
    pt = sum(r.prompt_token_count for r in rates)
    return AggregateRates(
        skill_hit_rate=skill_h / skill_t if skill_t > 0 else 0.0,
        non_skill_hit_rate=ns_h / ns_t if ns_t > 0 else 0.0,
        overall_hit_rate=total_h / pt if pt > 0 else 0.0,
        skill_hit_tokens=skill_h, non_skill_hit_tokens=ns_h, total_hit_tokens=total_h,
        skill_total_tokens=skill_t, non_skill_total_tokens=ns_t, prompt_token_count=pt,
        n_requests=len(rates),
    )
