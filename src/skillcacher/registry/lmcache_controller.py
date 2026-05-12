"""Wrapper over LMCache's Controller API for chunk-level Lookup + Pin.

LMCache exposes Controller.Lookup / Pin as an in-process API on the engine
side. the harness talks to it from the proxy via an HTTP shim that LMCache's
production-stack reference deployment ships, OR via the in-process Python
API when the proxy and engine share a process (offline dev only).

For the RunPod / remote-engine topology, we expose a tiny FastAPI shim on
the pod (see scripts/deploy/lmcache_shim.py installed by bootstrap.sh)
that wraps lmcache.v1.controller into HTTP endpoints. This wrapper is the
client side of that shim.

If the shim is unreachable, lookup() returns a 0-hit LookupResult with
error set, so the proxy continues without the chunk-level signal."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("skillcacher.lmcache_controller")


@dataclass
class LookupResult:
    hit_tokens: int
    total_tokens: int
    chunk_hits: int = 0
    total_chunks: int = 0
    error: str | None = None


class _HTTPShimClient:
    """Talks to the LMCache shim that bootstrap.sh runs alongside vLLM."""
    def __init__(self, base: str, key: str):
        self._base = base.rstrip("/")
        self._key = key
        self._headers = {"Authorization": f"Bearer {key}"} if key else {}

    async def lookup(self, token_ids: list[int]) -> dict:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(f"{self._base}/lookup", json={"token_ids": token_ids}, headers=self._headers)
            r.raise_for_status()
            return r.json()

    async def pin(self, token_ids: list[int], ttl: float | None) -> dict:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(f"{self._base}/pin", json={"token_ids": token_ids, "ttl": ttl}, headers=self._headers)
            r.raise_for_status()
            return r.json()


class LMCacheController:
    """Async wrapper over the LMCache Controller HTTP shim."""

    def __init__(self, api_url: str, api_key: str = ""):
        self.api_url = api_url
        self._client = _HTTPShimClient(api_url, api_key)

    async def lookup(self, token_ids: list[int]) -> LookupResult:
        try:
            d = await self._client.lookup(token_ids)
        except Exception as e:
            log.warning("lookup failed: %s", e)
            return LookupResult(hit_tokens=0, total_tokens=len(token_ids), error=str(e))
        return LookupResult(
            hit_tokens=d.get("hit_tokens", 0),
            total_tokens=d.get("total_tokens", len(token_ids)),
            chunk_hits=d.get("chunk_hits", 0),
            total_chunks=d.get("total_chunks", 0),
        )

    async def pin(self, token_ids: list[int], ttl: float | None = None) -> bool:
        try:
            await self._client.pin(token_ids, ttl)
            return True
        except Exception as e:
            log.warning("pin failed: %s", e)
            return False
