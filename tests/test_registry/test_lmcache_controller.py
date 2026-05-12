"""Tests for the LMCache Controller wrapper. We mock the underlying
lmcache.experimental.controller calls so tests stay offline."""
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from skillcacher.registry.lmcache_controller import (
    LMCacheController, LookupResult,
)


@pytest.mark.asyncio
async def test_lookup_returns_result():
    # Mock the underlying controller object the wrapper holds
    fake_inner = MagicMock()
    fake_inner.lookup = AsyncMock(return_value={"hit_tokens": 256, "total_tokens": 1024})
    c = LMCacheController(api_url="http://x", api_key="k")
    c._client = fake_inner
    res = await c.lookup([1, 2, 3, 4])
    assert isinstance(res, LookupResult)
    assert res.hit_tokens == 256
    assert res.total_tokens == 1024


@pytest.mark.asyncio
async def test_lookup_handles_no_hit():
    fake_inner = MagicMock()
    fake_inner.lookup = AsyncMock(return_value={"hit_tokens": 0, "total_tokens": 1024})
    c = LMCacheController(api_url="http://x", api_key="k")
    c._client = fake_inner
    res = await c.lookup([1, 2, 3, 4])
    assert res.hit_tokens == 0


@pytest.mark.asyncio
async def test_pin_invokes_inner():
    fake_inner = MagicMock()
    fake_inner.pin = AsyncMock(return_value={"ok": True})
    c = LMCacheController(api_url="http://x", api_key="k")
    c._client = fake_inner
    await c.pin([1, 2, 3], ttl=None)
    fake_inner.pin.assert_called_once()


@pytest.mark.asyncio
async def test_lookup_falls_back_on_exception():
    """If the controller call raises, lookup returns a 0-hit LookupResult
    so the proxy can continue without the chunk-level signal."""
    fake_inner = MagicMock()
    fake_inner.lookup = AsyncMock(side_effect=RuntimeError("boom"))
    c = LMCacheController(api_url="http://x", api_key="k")
    c._client = fake_inner
    res = await c.lookup([1, 2, 3])
    assert res.hit_tokens == 0
    assert res.error is not None
