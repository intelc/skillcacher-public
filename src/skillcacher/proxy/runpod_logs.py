"""Tail vLLM container logs via RunPod's REST API to get per-request
LMCache hit-token totals as ground truth.

Pinned LMCache log line shape (lmcache.integration.vllm.vllm_v1_adapter):
  LMCache INFO: Reqid: <id>, Total tokens N, LMCache hit tokens: M, need to load: K

Drift in this format breaks the per-request total path. test_runpod_logs
includes a regression test on the line shape.
"""
from __future__ import annotations

import asyncio
import re
import time
from typing import Any

import httpx

LMCACHE_LINE = re.compile(
    r"LMCache INFO:\s*Reqid:\s*([\w\-]+),\s*"
    r"Total tokens\s*(\d+),\s*"
    r"LMCache hit tokens:\s*(\d+)"
    r"(?:,\s*need to load:\s*(\d+))?"
)


def parse_lmcache_line(line: str) -> dict[str, Any] | None:
    m = LMCACHE_LINE.search(line)
    if not m:
        return None
    return {
        "reqid": m.group(1),
        "total_tokens": int(m.group(2)),
        "hit_tokens": int(m.group(3)),
        "need_to_load": int(m.group(4)) if m.group(4) else None,
    }


async def _fetch_logs_chunk(api_key: str, pod_id: str, since_cursor: str | None) -> dict:
    """Fetch a chunk of pod logs via RunPod REST API. Returns dict with
    `logs` (str) and optional `cursor` for resumption.

    NB: RunPod's logs endpoint returns the most recent chunk; for a real
    tail we accept duplication and let the parser dedupe by reqid."""
    url = f"https://rest.runpod.io/v1/pods/{pod_id}/logs"
    headers = {"Authorization": f"Bearer {api_key}"}
    params: dict[str, str] = {}
    if since_cursor:
        params["since"] = since_cursor
    async with httpx.AsyncClient(timeout=10.0) as client:
        r = await client.get(url, headers=headers, params=params)
        r.raise_for_status()
        return r.json()


async def tail_for_reqid(
    *, api_key: str, pod_id: str, target_reqid: str,
    timeout_s: float = 30.0, poll_interval_s: float = 1.0,
) -> dict[str, Any] | None:
    """Poll the pod logs endpoint until a line matching target_reqid is seen.
    Returns the parsed dict or None on timeout."""
    deadline = time.monotonic() + timeout_s
    cursor: str | None = None
    while time.monotonic() < deadline:
        try:
            chunk = await _fetch_logs_chunk(api_key, pod_id, cursor)
        except Exception:
            await asyncio.sleep(poll_interval_s)
            continue
        cursor = chunk.get("cursor")
        for line in (chunk.get("logs") or "").splitlines():
            parsed = parse_lmcache_line(line)
            if parsed and parsed["reqid"] == target_reqid:
                return parsed
        await asyncio.sleep(poll_interval_s)
    return None
