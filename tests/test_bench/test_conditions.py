"""Tests for per-condition pod lifecycle. Mock the subprocess + RunPod API
+ HTTP /health calls + local proxy spawn."""
import os
import subprocess
from unittest.mock import patch, MagicMock, AsyncMock
import pytest

from skillcacher.bench.conditions import ConditionLifecycle, Condition


def _completed(returncode: int, stdout: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["bash", "scripts/dev/oneshot_pod.sh"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


@pytest.mark.asyncio
async def test_lifecycle_runs_oneshot_spawns_proxy_then_delete():
    fake_run = MagicMock(
        return_value=_completed(
            returncode=0,
            stdout="POD_ID=abc\nPROXY_URL=https://abc-8000.proxy.runpod.net\n",
        )
    )
    fake_health = AsyncMock(return_value=True)
    fake_delete = MagicMock()
    fake_start_proxy = AsyncMock()
    with patch("skillcacher.bench.conditions._run_streaming", new=fake_run), \
         patch("skillcacher.bench.conditions.wait_health", new=fake_health), \
         patch("skillcacher.bench.conditions._delete_pod_by_name", new=fake_delete), \
         patch.object(ConditionLifecycle, "_start_local_proxy", new=fake_start_proxy):
        async with ConditionLifecycle(Condition("no_cache")) as lc:
            assert lc.proxy_url.endswith("proxy.runpod.net")
            assert lc.pod_id == "abc"
            # Cleanup-by-name uses the unique pod name (PID-suffixed).
            expected_pod_name = lc.pod_name
        cmds = [c.args[0] for c in fake_run.call_args_list]
        assert any("oneshot_pod.sh" in " ".join(c) for c in cmds), \
            f"expected oneshot_pod.sh call, got {cmds}"
        assert not any("provision.sh" in " ".join(c) for c in cmds)
        assert not any("bootstrap.sh" in " ".join(c) for c in cmds)
        assert fake_health.called
        assert fake_start_proxy.called
        fake_delete.assert_called_once_with(expected_pod_name)


@pytest.mark.asyncio
async def test_lifecycle_passes_condition_and_pod_name_via_env():
    """oneshot_pod.sh reads CONDITION and POD_NAME from the environment.
    POD_NAME includes the bench process PID so concurrent invocations don't
    collide on cleanup-by-name."""
    captured_envs = []

    def _capture(args, env=None, **kwargs):
        captured_envs.append(dict(env or {}))
        return _completed(returncode=0, stdout="POD_ID=x\nPROXY_URL=https://x.test\n")

    fake_health = AsyncMock(return_value=True)
    fake_delete = MagicMock()
    fake_start_proxy = AsyncMock()
    with patch("skillcacher.bench.conditions._run_streaming", side_effect=_capture), \
         patch("skillcacher.bench.conditions.wait_health", new=fake_health), \
         patch("skillcacher.bench.conditions._delete_pod_by_name", new=fake_delete), \
         patch.object(ConditionLifecycle, "_start_local_proxy", new=fake_start_proxy):
        async with ConditionLifecycle(Condition("cacheblend")):
            pass
    assert captured_envs, "_run_streaming was never called"
    env = captured_envs[0]
    assert env.get("CONDITION") == "cacheblend"
    pod_name = env.get("POD_NAME") or ""
    assert pod_name.startswith("skillcacher-bench-cacheblend-"), pod_name
    # Suffix is the current process PID.
    assert pod_name.endswith(f"-{os.getpid()}"), pod_name


@pytest.mark.asyncio
async def test_lifecycle_raises_on_oneshot_failure():
    fake_run = MagicMock(
        return_value=_completed(returncode=1, stdout="oneshot failed: see stderr above")
    )
    fake_delete = MagicMock()
    fake_start_proxy = AsyncMock()
    with patch("skillcacher.bench.conditions._run_streaming", new=fake_run), \
         patch("skillcacher.bench.conditions._delete_pod_by_name", new=fake_delete), \
         patch.object(ConditionLifecycle, "_start_local_proxy", new=fake_start_proxy):
        with pytest.raises(RuntimeError, match="oneshot_pod failed"):
            async with ConditionLifecycle(Condition("no_cache")):
                pass


@pytest.mark.asyncio
async def test_lifecycle_cleanup_on_raise_inside_aenter():
    """a raise inside __aenter__ (here: from _start_local_proxy
    after the pod has been provisioned) must DELETE the pod. Without the
    cleanup-on-raise wrapper, __aexit__ never runs and the pod leaks
    (~$0.10/min Llama-70B). 2026-05-09 hit cost ~$0.80 in two recoveries."""
    fake_run = MagicMock(
        return_value=_completed(
            returncode=0,
            stdout="POD_ID=zzz\nPROXY_URL=https://zzz-8000.proxy.runpod.net\n",
        )
    )
    fake_health = AsyncMock(return_value=True)
    fake_delete = MagicMock()
    fake_start_proxy = AsyncMock(side_effect=RuntimeError("local proxy never returned /health"))
    with patch("skillcacher.bench.conditions._run_streaming", new=fake_run), \
         patch("skillcacher.bench.conditions.wait_health", new=fake_health), \
         patch("skillcacher.bench.conditions._delete_pod_by_name", new=fake_delete), \
         patch.object(ConditionLifecycle, "_start_local_proxy", new=fake_start_proxy):
        lc = ConditionLifecycle(Condition("cacheblend"))
        expected_pod_name = lc.pod_name
        with pytest.raises(RuntimeError, match="local proxy never returned"):
            async with lc:
                pass
    fake_delete.assert_called_once_with(expected_pod_name)


@pytest.mark.asyncio
async def test_lifecycle_cleanup_on_oneshot_failure_also_deletes():
    """Even when oneshot_pod.sh itself fails (rc != 0), the cleanup-on-raise
    wrapper must still DELETE-by-name. Capacity-stuck attempts can leave a
    half-provisioned pod that responds to DELETE-by-name even though no
    POD_ID was ever parsed."""
    fake_run = MagicMock(
        return_value=_completed(returncode=1, stdout="oneshot failed: see stderr above")
    )
    fake_delete = MagicMock()
    fake_start_proxy = AsyncMock()
    with patch("skillcacher.bench.conditions._run_streaming", new=fake_run), \
         patch("skillcacher.bench.conditions._delete_pod_by_name", new=fake_delete), \
         patch.object(ConditionLifecycle, "_start_local_proxy", new=fake_start_proxy):
        lc = ConditionLifecycle(Condition("no_cache"))
        expected_pod_name = lc.pod_name
        with pytest.raises(RuntimeError, match="oneshot_pod failed"):
            async with lc:
                pass
    fake_delete.assert_called_once_with(expected_pod_name)
