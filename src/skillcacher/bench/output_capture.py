"""Quality-eval: replay a captured request through the proxy and
return the model's generated content.

Distinct from `workload.send_request`, which only records ack + summary
for the existing hit-rate-only bench harness. This path forces unary
(stream=False) so the proxy returns one JSON body and we don't have to
reassemble SSE deltas just to compare outputs."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import httpx


@dataclass
class Generation:
    text: str
    content_blocks: list[dict]
    stop_reason: str
    input_tokens: int
    output_tokens: int
    response_id: str
    ttft_ms: float
    raw_response: dict | None = None


def _extract_text(content_blocks: list[dict]) -> str:
    return "".join(b.get("text", "") for b in content_blocks if b.get("type") == "text")


async def replay_with_output_capture(
    req_body: dict,
    settings,
    *,
    temperature: float | None = None,
    proxy_url: str | None = None,
    client: httpx.AsyncClient | None = None,
) -> Generation:
    """POST a captured request body through the proxy and parse the response.

    Forces stream=False so we get a single Anthropic-format JSON response.
    If `temperature` is supplied it overrides whatever was in req_body —
    captured CC requests usually pin temperature, and we need to swap
    between T=0 and T=0.7 for §1 vs §2 without re-capturing fixtures.

    `proxy_url` and `client` are injectable for tests; in production both
    fall back to the settings-derived URL and a fresh AsyncClient."""
    body = copy.deepcopy(req_body)
    body["stream"] = False
    if temperature is not None:
        body["temperature"] = temperature
    url = proxy_url or f"http://{settings.proxy_host}:{settings.proxy_port}/v1/messages"

    t0 = time.time()
    if client is not None:
        r = await client.post(url, json=body)
    else:
        async with httpx.AsyncClient(timeout=settings.request_timeout_s) as c:
            r = await c.post(url, json=body)
    ttft_ms = (time.time() - t0) * 1000.0
    r.raise_for_status()
    raw = r.json()

    content = raw.get("content") or []
    usage = raw.get("usage") or {}
    return Generation(
        text=_extract_text(content),
        content_blocks=content,
        stop_reason=raw.get("stop_reason") or "",
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        response_id=raw.get("id") or "",
        ttft_ms=ttft_ms,
        raw_response=raw,
    )
