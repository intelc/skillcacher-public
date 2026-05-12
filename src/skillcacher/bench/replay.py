"""Bench replay with chunk-level join.

Replaces the harness's proportional attribution. The numerator for
skill_hit_rate comes directly from Controller.Lookup chunk-level results,
joined by token range to span tags."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from skillcacher.proxy.trace_store import TraceStore, RequestRecord


@dataclass
class DecomposedRates:
    skill_hit_rate: float
    non_skill_hit_rate: float
    overall_hit_rate: float
    skill_hit_tokens: int
    non_skill_hit_tokens: int
    total_hit_tokens: int
    skill_total_tokens: int
    non_skill_total_tokens: int
    prompt_token_count: int


def compute_decomposed_hit_rate(
    record: RequestRecord, *, span_id_to_tag: dict[str, str],
) -> DecomposedRates:
    """Compute per-record decomposed hit rates from chunk-level Lookup data.

    span_id_to_tag maps each registered span to its tag (`skill_body`,
    `tool_def`, `system_static`, `dynamic`, `other`)."""
    skill_hit_tokens = sum(
        l.hit_tokens for l in record.span_lookups
        if span_id_to_tag.get(l.span_id) == "skill_body"
    )
    if record.engine_total_hit_tokens is not None:
        total_hit_tokens = record.engine_total_hit_tokens
    else:
        total_hit_tokens = sum(l.hit_tokens for l in record.span_lookups)
    non_skill_hit_tokens = max(0, total_hit_tokens - skill_hit_tokens)
    skill_total = sum(
        l.total_tokens for l in record.span_lookups
        if span_id_to_tag.get(l.span_id) == "skill_body"
    )
    non_skill_total = max(0, record.prompt_token_count - skill_total)
    return DecomposedRates(
        skill_hit_rate=skill_hit_tokens / skill_total if skill_total > 0 else 0.0,
        non_skill_hit_rate=non_skill_hit_tokens / non_skill_total if non_skill_total > 0 else 0.0,
        overall_hit_rate=total_hit_tokens / record.prompt_token_count if record.prompt_token_count > 0 else 0.0,
        skill_hit_tokens=skill_hit_tokens,
        non_skill_hit_tokens=non_skill_hit_tokens,
        total_hit_tokens=total_hit_tokens,
        skill_total_tokens=skill_total,
        non_skill_total_tokens=non_skill_total,
        prompt_token_count=record.prompt_token_count,
    )


def replay_dir(trace_dir: Path, *, span_id_to_tag: dict[str, str]) -> list[DecomposedRates]:
    """Iterate a trace directory and produce per-request DecomposedRates."""
    store = TraceStore(trace_dir)
    return [compute_decomposed_hit_rate(rec, span_id_to_tag=span_id_to_tag) for rec in store.read_all()]
