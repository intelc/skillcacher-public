"""pre-seed wiring. Verify the bench's per-condition local proxy
spawn passes SKILLCACHER_ENABLE_PRE_SEED=true for every condition, so the
proxy startup hook in proxy/server.py calls pre_seed_skills.

Pre-seed runs for all three conditions (no_cache included) for cross-
condition parity — comparing pre-seeded cacheblend vs cold prefix_cache
would be apples-to-oranges. Lookup stays off because the lmcache_shim
isn't launched on oneshot pods (per the harness design §1)."""
import os
import subprocess
from unittest.mock import patch, MagicMock, AsyncMock

import pytest

from skillcacher.bench.conditions import ConditionLifecycle, Condition


def _completed(returncode: int = 0, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["bash", "scripts/dev/oneshot_pod.sh"],
        returncode=returncode, stdout=stdout, stderr="",
    )


class _FakePopen:
    """Captures the env dict passed at spawn time. Class attribute so the
    test can read it after the lifecycle context exits."""
    last_env: dict | None = None

    def __init__(self, args, env=None, **kw):
        type(self).last_env = dict(env or {})
        self.pid = 12345
        self._rc: int | None = None

    def poll(self):
        return self._rc

    def terminate(self):
        self._rc = 0

    def kill(self):
        self._rc = -9

    def wait(self, timeout=None):
        self._rc = 0
        return 0


def _ok_async_client(*a, **kw):
    """Async context manager whose .get() and .post() always return 200.
    Stands in for httpx.AsyncClient inside _start_local_proxy."""
    resp = MagicMock(status_code=200, text="{}")
    client = MagicMock()
    client.get = AsyncMock(return_value=resp)
    client.post = AsyncMock(return_value=resp)

    class _CM:
        async def __aenter__(self_inner):
            return client

        async def __aexit__(self_inner, *_):
            return False

    return _CM()


@pytest.mark.parametrize("condition", ["no_cache", "prefix_cache", "cacheblend"])
@pytest.mark.asyncio
async def test_preseed_env_is_true_for_every_condition(condition, tmp_path, monkeypatch):
    """The bench's per-condition local proxy spawn must set
    SKILLCACHER_ENABLE_PRE_SEED=true so the proxy startup hook in
    server.py runs pre_seed_skills against the pod that was just brought
    up. the harness explicitly disabled this; the harness flips it back on."""
    monkeypatch.setenv("SKILLCACHER_TRACE_DIR", str(tmp_path / condition / "traces"))
    _FakePopen.last_env = None

    fake_oneshot = MagicMock(
        return_value=_completed(0, f"POD_ID=test-{condition}\nPROXY_URL=https://t.test\n")
    )

    with patch("skillcacher.bench.conditions._run_streaming", new=fake_oneshot), \
         patch("skillcacher.bench.conditions.wait_health", new=AsyncMock(return_value=True)), \
         patch("skillcacher.bench.conditions._delete_pod_by_name", new=MagicMock()), \
         patch("skillcacher.bench.conditions.subprocess.Popen", new=_FakePopen), \
         patch("skillcacher.bench.conditions.httpx.AsyncClient", new=_ok_async_client):
        async with ConditionLifecycle(Condition(condition)):
            pass

    assert _FakePopen.last_env is not None, "local proxy spawn was never reached"
    env = _FakePopen.last_env
    assert env.get("SKILLCACHER_ENABLE_PRE_SEED") == "true", \
        f"{condition}: expected pre-seed=true, got {env.get('SKILLCACHER_ENABLE_PRE_SEED')!r}"
    # Lookup stays off until the lmcache_shim is launched on oneshot pods.
    # Pre-seed without lookup is the documented the harness path: warmup
    # prefill populates lmcache via vLLM's kv_transfer; pin step skipped.
    assert env.get("SKILLCACHER_ENABLE_LOOKUP") == "false", \
        f"{condition}: expected lookup=false (no shim), got {env.get('SKILLCACHER_ENABLE_LOOKUP')!r}"
