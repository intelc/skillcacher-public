"""Scrape LMCache aggregate counters from vLLM's /metrics endpoint.

Used as the bench-run-level sanity bound for chunk-level Lookup attribution."""
from __future__ import annotations

import re
from dataclasses import dataclass

import httpx

LMCACHE_METRIC_NAMES = {
    "lmcache:num_hit_tokens",
    "lmcache:num_lookup_hits",
    "lmcache:num_vllm_hit_tokens",
    "lmcache:retrieve_hit_rate",
    "lmcache:lookup_hit_rate",
}

METRIC_LINE = re.compile(r"^([\w:]+)(?:\{[^}]*\})?\s+([0-9.\-+eE]+)\s*$")


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = METRIC_LINE.match(line)
        if not m:
            continue
        name, value = m.group(1), m.group(2)
        if name in LMCACHE_METRIC_NAMES:
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


@dataclass
class MetricsDelta:
    delta: dict[str, float]

    @classmethod
    def from_pair(cls, before: dict[str, float], after: dict[str, float]) -> "MetricsDelta":
        return cls(delta={k: after.get(k, 0.0) - before.get(k, 0.0) for k in set(before) | set(after)})


async def snapshot(api_base: str) -> dict[str, float]:
    """Fetch /metrics from a vLLM endpoint and return parsed LMCache metrics.

    Custom User-Agent: RunPod's Cloudflare layer 403s default httpx/urllib UA
    on `*.proxy.runpod.net`. Required for any probe that goes through the
    public proxy URL."""
    url = api_base.rstrip("/") + "/metrics"
    headers = {"User-Agent": "skillcacher-metrics/0.1"}
    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        r = await client.get(url)
        r.raise_for_status()
        return parse_prometheus_metrics(r.text)
