import pytest
from unittest.mock import patch, AsyncMock

from skillcacher.proxy.lmcache_metrics import (
    parse_prometheus_metrics, snapshot, MetricsDelta,
)


SAMPLE_METRICS = """\
# HELP lmcache:num_hit_tokens Total number of cache hit tokens from retrieval
# TYPE lmcache:num_hit_tokens counter
lmcache:num_hit_tokens 12345
# HELP lmcache:num_lookup_hits Total number of tokens hit in lookup operations
# TYPE lmcache:num_lookup_hits counter
lmcache:num_lookup_hits 67890
# HELP lmcache:retrieve_hit_rate The hit rate for retrieve requests
# TYPE lmcache:retrieve_hit_rate gauge
lmcache:retrieve_hit_rate 0.85
"""


def test_parse_prometheus_metrics_extracts_counters_and_gauges():
    out = parse_prometheus_metrics(SAMPLE_METRICS)
    assert out["lmcache:num_hit_tokens"] == 12345.0
    assert out["lmcache:num_lookup_hits"] == 67890.0
    assert out["lmcache:retrieve_hit_rate"] == 0.85


def test_metrics_delta_arithmetic():
    a = {"x": 100.0, "y": 5.0}
    b = {"x": 130.0, "y": 5.0}
    d = MetricsDelta.from_pair(a, b)
    assert d.delta["x"] == 30.0
    assert d.delta["y"] == 0.0


@pytest.mark.asyncio
async def test_snapshot_fetches_metrics():
    fake_get = AsyncMock(return_value=type("R", (), {
        "raise_for_status": lambda self: None,
        "text": SAMPLE_METRICS,
    })())
    with patch("httpx.AsyncClient.get", new=fake_get):
        snap = await snapshot("http://pod:8000")
    assert snap["lmcache:num_hit_tokens"] == 12345.0
