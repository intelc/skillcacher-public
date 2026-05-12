"""Extract per-request hit-rate metrics from a vllm.log captured during a
condition's run.

Why this exists: the lmcache cu12 image's vllm exposes only `vllm:*`
counters via /metrics — no `lmcache:*` prefixed counters and no per-request
hit_tokens in the response body. The hit information is only present in
log lines like:

    LMCache INFO: Reqid: chatcmpl-XXX, Total tokens 1856, Inference Engine
    computed tokens: 1840, LMCache hit tokens: 1721, need to load: 0

`ConditionLifecycle.__aexit__` SSHes into the pod and saves vllm.log to
`<run_root>/<condition>/vllm.log` before deleting the pod. This module
parses that file. Followup #6 close.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from skillcacher.bench.metrics import AggregateRates


_HIT_LINE_RE = re.compile(
    r"Reqid:\s*(?P<req_id>\S+?),\s*"
    r"Total tokens\s+(?P<total>\d+),\s*"
    r"Inference Engine computed tokens:\s*(?P<computed>\d+),\s*"
    r"LMCache hit tokens:\s*(?P<hit>\d+)"
)


@dataclass
class PerRequestHits:
    req_id: str
    total_tokens: int
    computed_tokens: int  # vllm's own prefix-cache hit count
    hit_tokens: int       # LMCache's hit count (cacheblend or prefix-only)


def parse_vllm_log_for_hits(log_path: Path) -> list[PerRequestHits]:
    """Iterate vllm.log; return one PerRequestHits per matching line.

    Each request typically produces ONE such line (the lookup result line
    from `vllm_v1_adapter.py`). Order in the file matches request order."""
    out: list[PerRequestHits] = []
    if not log_path.exists():
        return out
    seen_req_ids: set[str] = set()
    for line in log_path.read_text(errors="replace").splitlines():
        m = _HIT_LINE_RE.search(line)
        if not m:
            continue
        req_id = m.group("req_id").rstrip(",")
        # Some lmcache versions log multiple times per request as the engine
        # progresses through layerwise prefill; first-write-wins gives the
        # initial lookup result which is what we care about.
        if req_id in seen_req_ids:
            continue
        seen_req_ids.add(req_id)
        out.append(PerRequestHits(
            req_id=req_id,
            total_tokens=int(m.group("total")),
            computed_tokens=int(m.group("computed")),
            hit_tokens=int(m.group("hit")),
        ))
    return out


def aggregate_log_hits(per_request: list[PerRequestHits]) -> AggregateRates:
    """Roll per-request hits into the AggregateRates shape that write_report
    expects. We don't have skill/non-skill decomposition (warmup pre-seed
    not wired — T34.5 phase 2), so skill_* fields stay 0."""
    if not per_request:
        return AggregateRates(0.0, 0.0, 0.0, 0, 0, 0, 0, 0, 0, 0)
    total_hit = sum(r.hit_tokens for r in per_request)
    total_prompt = sum(r.total_tokens for r in per_request)
    return AggregateRates(
        skill_hit_rate=0.0,
        non_skill_hit_rate=total_hit / total_prompt if total_prompt > 0 else 0.0,
        overall_hit_rate=total_hit / total_prompt if total_prompt > 0 else 0.0,
        skill_hit_tokens=0,
        non_skill_hit_tokens=total_hit,
        total_hit_tokens=total_hit,
        skill_total_tokens=0,
        non_skill_total_tokens=total_prompt,
        prompt_token_count=total_prompt,
        n_requests=len(per_request),
    )
