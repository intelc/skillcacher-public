"""Replay a captured fixture through the live proxy."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

import httpx


def iter_requests(fixture_set: str, workload: str, start_at: int = 0) -> Iterator[tuple[int, dict]]:
    """Yield (idx, request_body_dict) from a captured fixture's per-request JSON.

    Fixtures live under tests/fixtures/<fixture_set>/<task_id>/raw_requests/<idx>.json
    (created by a captured trace). For now we keep the loader simple and
    require the fixtures to already be in this shape."""
    root = Path("tests/fixtures") / fixture_set / workload / "raw_requests"
    if not root.exists():
        return
    for path in sorted(root.glob("*.json")):
        idx = int(path.stem)
        if idx < start_at:
            continue
        yield idx, json.loads(path.read_text())


async def send_request(req_body: dict, settings) -> dict:
    """POST a captured request body through the local proxy. Returns a
    minimal record dict (the proxy writes the full RequestRecord to the
    trace store independently; this just records ack + summary for the
    checkpoint store)."""
    proxy_url = f"http://{settings.proxy_host}:{settings.proxy_port}/v1/messages"
    async with httpx.AsyncClient(timeout=600.0) as client:
        if req_body.get("stream"):
            # Drain the stream
            text = ""
            async with client.stream("POST", proxy_url, json=req_body) as r:
                async for line in r.aiter_lines():
                    text += line + "\n"
            return {"status": r.status_code, "stream_chars": len(text)}
        r = await client.post(proxy_url, json=req_body)
        return {"status": r.status_code, "body_chars": len(r.text)}
