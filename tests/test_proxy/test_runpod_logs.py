"""Tests for runpod_logs: parsing LMCache stdout lines for per-request
hit-tokens, and a tail-with-timeout helper that polls the RunPod logs API."""
import pytest
from unittest.mock import patch, AsyncMock

from skillcacher.proxy.runpod_logs import parse_lmcache_line, tail_for_reqid


SAMPLE_LINE = (
    "[2025-08-04 22:43:35,484] LMCache INFO: Reqid: chatcmpl-05e2d296601046b29210f53a1fa30b13, "
    "Total tokens 1536, LMCache hit tokens: 1536, need to load: 1535 "
    "(vllm_v1_adapter.py:803:lmcache.integration.vllm.vllm_v1_adapter)"
)


def test_parse_lmcache_line_extracts_fields():
    out = parse_lmcache_line(SAMPLE_LINE)
    assert out is not None
    assert out["reqid"] == "chatcmpl-05e2d296601046b29210f53a1fa30b13"
    assert out["total_tokens"] == 1536
    assert out["hit_tokens"] == 1536
    assert out["need_to_load"] == 1535


def test_parse_lmcache_line_returns_none_on_unrelated():
    assert parse_lmcache_line("vllm INFO: server started") is None


def test_parse_lmcache_line_handles_no_load_field():
    line = "[ts] LMCache INFO: Reqid: foo, Total tokens 10, LMCache hit tokens: 5"
    out = parse_lmcache_line(line)
    assert out["reqid"] == "foo"
    assert out["hit_tokens"] == 5
    assert out["need_to_load"] is None


@pytest.mark.asyncio
async def test_tail_for_reqid_returns_match():
    logs_seq = [
        "irrelevant line\n",
        SAMPLE_LINE + "\n",
    ]
    fake_get = AsyncMock(side_effect=[
        {"logs": "irrelevant line\n"},
        {"logs": SAMPLE_LINE + "\n"},
    ])
    with patch("skillcacher.proxy.runpod_logs._fetch_logs_chunk", new=fake_get):
        result = await tail_for_reqid(
            api_key="k", pod_id="p",
            target_reqid="chatcmpl-05e2d296601046b29210f53a1fa30b13",
            timeout_s=1.0, poll_interval_s=0.01,
        )
    assert result is not None
    assert result["hit_tokens"] == 1536


@pytest.mark.asyncio
async def test_tail_for_reqid_times_out_returns_none():
    fake_get = AsyncMock(return_value={"logs": "nothing here\n"})
    with patch("skillcacher.proxy.runpod_logs._fetch_logs_chunk", new=fake_get):
        result = await tail_for_reqid(
            api_key="k", pod_id="p",
            target_reqid="never-comes",
            timeout_s=0.1, poll_interval_s=0.01,
        )
    assert result is None
