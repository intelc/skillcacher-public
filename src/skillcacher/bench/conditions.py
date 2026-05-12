"""Per-condition pod lifecycle: provision via oneshot_pod.sh → wait /health
→ DELETE-by-name on exit.

History (T34.5): originally called `scripts/deploy/{provision,bootstrap}.sh`,
but bootstrap.sh lacks the cacheblend 7-patch recipe (PR LMCache#2946 fixes)
that landed in `scripts/dev/oneshot_pod.{sh,py}` during T24 unblocking. The
`cacheblend` condition would silently degrade to plain prefix_cache without
the patches. Migrated to oneshot for T35 onward; full retirement of the old
path is gated on Phase 2 of T34.5 (lmcache_shim + Llama tuning + version
pinning brought into oneshot before canonical run).

KNOWN GAP: oneshot_pod.sh does NOT launch `scripts/deploy/lmcache_shim.py`,
so `registry/warmup.py` and `Controller.Lookup` calls in `proxy/server.py`
have no shim to talk to. The proxy's `SKILLCACHER_ENABLE_PRE_SEED=false`
default keeps this safe — pre-seed becomes a no-op. The bench's `cacheblend`
condition still produces real cacheblend hits via lmcache's segment-based
content lookup; only proxy-side skill pre-seed is degraded. Phase 2 of
T34.5 closes this gap.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import urllib.parse
import urllib.request
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from pathlib import Path

import httpx

log = logging.getLogger("skillcacher.conditions")


@dataclass
class Condition:
    name: str  # one of: no_cache, prefix_cache, cacheblend


async def wait_health(proxy_url: str, timeout_s: float = 600.0) -> bool:
    """Poll /health until 200 or timeout. oneshot_pod.sh already waits for
    /health internally before printing PROXY_URL, so this is mostly a sanity
    re-check; we keep it for resilience to transient Cloudflare blips."""
    deadline = asyncio.get_event_loop().time() + timeout_s
    headers = {"User-Agent": "skillcacher-bench/0.1"}  # avoid Cloudflare 403
    while asyncio.get_event_loop().time() < deadline:
        try:
            async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
                r = await client.get(f"{proxy_url}/health")
                if r.status_code == 200:
                    return True
        except Exception:
            pass
        await asyncio.sleep(5)
    return False


def _run_streaming(
    args: list[str],
    *,
    env: dict[str, str] | None = None,
    log_prefix: str = "[subprocess]",
) -> subprocess.CompletedProcess:
    """Like ``subprocess.run(capture_output=True, text=True)`` but stream each
    stdout line to ``log.info`` as it arrives.

    Default ``subprocess.run(capture_output=True)`` buffers all output until
    the process exits, so callers see nothing during long pod boots — we were
    blind for 30+ minutes during multiple attempts. This wrapper emits
    every line live while still capturing the full output for the caller's
    parsing logic. stderr is merged into stdout so the streaming order is
    preserved.

    Returns a CompletedProcess with the same shape as ``subprocess.run``
    (stdout populated, stderr empty since merged).
    """
    proc = subprocess.Popen(  # noqa: S603 — caller controls argv
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip("\n")
        captured.append(line)
        if line:
            log.info("%s %s", log_prefix, line)
    rc = proc.wait()
    return subprocess.CompletedProcess(
        args=args, returncode=rc, stdout="\n".join(captured), stderr=""
    )


def _delete_pod_by_name(pod_name: str) -> None:
    """DELETE every RunPod pod with this exact name. Mirrors the cleanup
    pattern in scripts/dev/cacheblend_proof.sh — by-name (not by captured
    POD_ID variable) so a partially-failed provision still tears down."""
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        log.warning("RUNPOD_API_KEY missing; cannot delete pod %s", pod_name)
        return
    qs = urllib.parse.urlencode({"name": pod_name})
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods?{qs}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read())
    except Exception as e:
        log.warning("pod lookup failed for %s: %s", pod_name, e)
        return
    pods = data if isinstance(data, list) else (data or {}).get("pods", [])
    for p in pods:
        if p.get("name") != pod_name:
            continue
        pid = p["id"]
        d_req = urllib.request.Request(
            f"https://rest.runpod.io/v1/pods/{pid}",
            method="DELETE",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        try:
            with urllib.request.urlopen(d_req, timeout=30) as dr:
                log.info("deleted pod %s ('%s') status=%s", pid, pod_name, dr.status)
        except Exception as e:
            log.warning("DELETE failed for pod %s: %s", pid, e)


def _get_pod_ssh_info(pod_id: str) -> tuple[str, int] | None:
    """Query the RunPod API for a pod's public IP + SSH port (mapped to 22).
    Returns (host, port) or None if unavailable. Used by the lifecycle exit
    to SSH in and dump vllm.log before the pod gets deleted."""
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key or not pod_id:
        return None
    req = urllib.request.Request(
        f"https://rest.runpod.io/v1/pods/{pod_id}",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            d = json.loads(r.read())
    except Exception as e:
        log.warning("pod %s SSH lookup failed: %s", pod_id, e)
        return None
    host = d.get("publicIp")
    port = (d.get("portMappings") or {}).get("22")
    if not host or not port:
        return None
    return host, int(port)


def _ssh_cat(ssh_host: str, ssh_port: int, ssh_key: str, remote_path: str,
             dest_path: Path, *, timeout: int = 60) -> bool:
    """SSH cat a remote file to dest_path. Returns True on rc=0 + non-empty.
    Best-effort — failures log a warning and return False."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ssh", "-i", ssh_key,
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "IdentitiesOnly=yes",
        "-o", "ConnectTimeout=15",
        "-p", str(ssh_port), f"root@{ssh_host}",
        f"cat {remote_path}",
    ]
    try:
        with dest_path.open("wb") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=timeout)
        size = dest_path.stat().st_size if dest_path.exists() else 0
        if r.returncode != 0 or size == 0:
            log.warning(
                "scp %s returned rc=%s size=%dB: %s",
                remote_path, r.returncode, size,
                r.stderr.decode(errors="replace")[:300],
            )
            return False
        log.info("dumped %s → %s (%d bytes)", remote_path, dest_path, size)
        return True
    except Exception as e:
        log.warning("scp %s failed: %s", remote_path, e)
        return False


def _ssh_dump_vllm_log(ssh_host: str, ssh_port: int, dest_path: Path,
                       ssh_key: str | None = None) -> bool:
    """SSH into the pod and dump /var/log/vllm.log to dest_path; ALSO dump
    /var/log/oneshot_boot.log to dest_path's sibling `oneshot_boot.log`.

    The boot log holds the `[patch] ...` lines that confirm whether the
    cacheblend dim-fix patch (and the others) actually applied. Without it
    we can't distinguish patch-not-applied from genuine retrieval failure.
    Returns True if vllm.log fetched (boot.log is best-effort)."""
    key = ssh_key or os.path.expanduser("~/.ssh/runpod_ed25519")
    if not Path(key).exists():
        alt = os.path.expanduser("~/.ssh/id_ed25519")
        if Path(alt).exists():
            key = alt
    ok = _ssh_cat(ssh_host, ssh_port, key, "/var/log/vllm.log", dest_path)
    boot_path = dest_path.parent / "oneshot_boot.log"
    _ssh_cat(ssh_host, ssh_port, key, "/var/log/oneshot_boot.log", boot_path)
    return ok


class ConditionLifecycle(AbstractAsyncContextManager):
    """Async context manager: enter brings up the pod for `condition` via
    oneshot_pod.sh AND spawns a local skillcacher-proxy pointed at it, exit
    optionally SCPs vllm.log out, kills the proxy, and DELETEs the pod.

    The local proxy is per-condition because skillcacher-proxy reads
    `SKILLCACHER_BACKEND_URL` at startup, not per-request — swapping
    conditions requires a fresh proxy with the new pod URL.

    `log_dump_path` (optional): if provided, SSH-dump /var/log/vllm.log to
    this path before the pod is deleted. Required for hit-rate metrics
    extraction (followup #6) since vllm doesn't expose `lmcache:*` counters
    via /metrics on the cu12 image."""

    def __init__(self, condition: Condition, *, log_dump_path: Path | None = None):
        self.condition = condition
        self.proxy_url: str = ""
        self.pod_id: str = ""
        # POD_NAME folds in the condition AND a per-process suffix so concurrent
        # bench invocations don't collide on cleanup-by-name. (Hit during
        #   a killed wrapper's EXIT trap deleted the
        # next wrapper's freshly-created pod when both used the same name.)
        self.pod_name = f"skillcacher-bench-{condition.name}-{os.getpid()}"
        self._proxy_proc: subprocess.Popen | None = None
        self._proxy_log_fh = None  # file handle for proxy stdout/stderr capture
        self.log_dump_path = log_dump_path

    async def __aenter__(self):
        try:
            log.info("provisioning condition=%s via oneshot_pod.sh", self.condition.name)
            env = os.environ.copy()
            env["CONDITION"] = self.condition.name
            env["POD_NAME"] = self.pod_name
            result = _run_streaming(
                ["bash", "scripts/dev/oneshot_pod.sh"],
                env=env, log_prefix=f"[oneshot:{self.condition.name}]",
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"oneshot_pod failed for {self.condition.name}: {result.stdout[-2000:]}"
                )
            for line in result.stdout.splitlines():
                if line.startswith("POD_ID="):
                    self.pod_id = line.split("=", 1)[1].strip()
                elif line.startswith("PROXY_URL="):
                    self.proxy_url = line.split("=", 1)[1].strip()
            if not self.proxy_url:
                raise RuntimeError(
                    f"no PROXY_URL emitted by oneshot_pod.sh for {self.condition.name}"
                )

            log.info("waiting for pod /health on %s (sanity recheck)", self.proxy_url)
            ok = await wait_health(self.proxy_url)
            if not ok:
                raise RuntimeError(
                    f"pod /health never returned 200 for {self.condition.name} "
                    f"(oneshot reported it ready — likely Cloudflare blip)"
                )

            await self._start_local_proxy()
            return self
        except BaseException:
            # Without this, a raise inside __aenter__ skips __aexit__, leaking
            # the pod (~$0.10/min Llama-70B) and any spawned local proxy.
            # Tier-3 capture; cost two manual recoveries.
            await self._cleanup()
            raise

    async def _start_local_proxy(self) -> None:
        """Spawn a local skillcacher-proxy pointed at this condition's pod.
        Per-condition because skillcacher-proxy reads its backend URL at
        startup, not per-request.

        Bails out early if 127.0.0.1:<proxy_port> is ALREADY bound by
        another process — otherwise our spawn silently fails (uvicorn exits
        on bind error, but our /health check passes against the *existing*
        process pointing at the wrong/dead pod, sending every bench
        request 502). a leftover proxy
        from a 6:54PM dev session was bound to 4000 and intercepted all
        bench requests. Burned a full $1.50 bench run before noticed."""
        # Read settings AT spawn time so the test suite's monkeypatched values
        # (and any ad-hoc env overrides) are honoured.
        from skillcacher.settings import Settings
        s = Settings()
        # Pre-spawn port check: bail loudly if something already owns the port.
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            try:
                probe.connect((s.proxy_host, s.proxy_port))
                already_bound = True
            except OSError:
                already_bound = False
        if already_bound:
            raise RuntimeError(
                f"port {s.proxy_host}:{s.proxy_port} already bound — likely a "
                f"leftover skillcacher-proxy from a previous session. Run "
                f"`lsof -i:{s.proxy_port}` and kill the orphan before retrying."
            )
        # Tokenizer name follows the served model; SKILLCACHER_TRACE_DIR is
        # per-condition so each condition's traces don't collide.
        trace_dir = os.path.abspath(
            os.environ.get("SKILLCACHER_TRACE_DIR", f"benchmark/results/_traces_{self.condition.name}")
        )
        os.makedirs(trace_dir, exist_ok=True)
        env = os.environ.copy()
        # Inherit MODEL_NAME (the same env var oneshot_pod.sh uses to select
        # the served model) so the proxy and pod agree on which model to
        # serve. bench was set with
        # MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct but the proxy stayed
        # on the Qwen3-8B default → vllm responded 404 to every request.
        backend_model = (
            env.get("SKILLCACHER_BACKEND_MODEL")
            or env.get("MODEL_NAME")
            or "Qwen/Qwen3-8B"
        )
        tokenizer = (
            env.get("SKILLCACHER_TOKENIZER_NAME")
            or env.get("MODEL_NAME")
            or "Qwen/Qwen3-8B"
        )
        env.update({
            "SKILLCACHER_BACKEND_URL": self.proxy_url,
            "SKILLCACHER_BACKEND_MODEL": backend_model,
            "SKILLCACHER_TOKENIZER_NAME": tokenizer,
            "SKILLCACHER_TRACE_DIR": trace_dir,
            # pre-seed runs against vllm directly via warmup_via_litellm.
            # The lmcache_shim is not launched on oneshot pods, so Controller is
            # None and the pin step is skipped — the warmup prefill alone populates
            # lmcache via vLLM's built-in kv_transfer integration. Lookup + stdout
            # tail remain off (no shim to talk to); per-request hit metrics come
            # from the vllm.log scrape path in bench/log_metrics.py instead.
            "SKILLCACHER_ENABLE_PRE_SEED": "true",
            "SKILLCACHER_ENABLE_LOOKUP": "false",
            "SKILLCACHER_ENABLE_STDOUT_TAIL": "false",
        })
        # Capture proxy stdout/stderr to a per-condition log file. Previously
        # routed to DEVNULL — left us blind during  
        # when call_unary raised on every request and we had no record of why.
        # Path mirrors vllm.log location: <run_root>/<cond>/proxy.log.
        proxy_log_path = Path(trace_dir).parent / "proxy.log"
        proxy_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._proxy_log_fh = proxy_log_path.open("ab", buffering=0)
        log.info("spawning local skillcacher-proxy → %s (trace_dir=%s, log=%s)",
                 self.proxy_url, trace_dir, proxy_log_path)
        self._proxy_proc = subprocess.Popen(
            [".venv/bin/skillcacher-proxy"],
            env=env,
            stdout=self._proxy_log_fh,
            stderr=subprocess.STDOUT,
        )
        # Wait for local /health. Pre-seed warmup runs at proxy startup
        # before /health responds; ~12 skills × 3 anchors × ~2-4K tokens
        # prefill each on Llama-70B can take 60-90s before /health flips.
        # Tunable via SKILLCACHER_PROXY_HEALTH_TIMEOUT_S (default 180).
        local_url = f"http://{s.proxy_host}:{s.proxy_port}"
        health_timeout_s = float(
            os.environ.get("SKILLCACHER_PROXY_HEALTH_TIMEOUT_S", "180")
        )
        deadline = asyncio.get_event_loop().time() + health_timeout_s
        while asyncio.get_event_loop().time() < deadline:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    r = await client.get(f"{local_url}/health")
                    if r.status_code == 200:
                        log.info("local proxy ready at %s", local_url)
                        await self._preflight_call_unary(local_url)
                        return
            except Exception:
                pass
            if self._proxy_proc.poll() is not None:
                raise RuntimeError(
                    f"local proxy exited early (rc={self._proxy_proc.returncode}); "
                    f"see {proxy_log_path} for tracebacks"
                )
            await asyncio.sleep(1)
        raise RuntimeError(
            f"local proxy never returned /health 200 within {health_timeout_s:.0f}s"
        )

    async def _preflight_call_unary(self, local_url: str) -> None:
        """One tiny POST to /v1/messages so we fail fast if the proxy →
        backend path is broken (auth, URL shape, Cloudflare). Hit during
          bench burned through full 5-request workload
        × 3 conditions, each request returning 502, before we noticed —
        because /health passing only proves the proxy is alive, not that it
        can actually reach the backend.

        Failure includes the body so the wrapper script's stdout shows the
        exact error, not just 'preflight failed'."""
        body = {
            "model": "preflight",
            "max_tokens": 4,
            "messages": [{"role": "user", "content": "ping"}],
        }
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                r = await client.post(f"{local_url}/v1/messages", json=body)
        except Exception as e:
            raise RuntimeError(f"preflight request raised: {e!r}") from e
        if r.status_code != 200:
            raise RuntimeError(
                f"preflight POST /v1/messages returned {r.status_code}: "
                f"{r.text[:500]}"
            )
        log.info("preflight call_unary ok (%d bytes)", len(r.text))

    async def _cleanup(self) -> None:
        """Tear down everything __aenter__ may have set up. Safe to call twice
        and safe to call when only some state was initialized — used by both
        __aexit__ (normal path) and the cleanup-on-raise wrapper in __aenter__."""
        # Kill the local proxy first so it doesn't keep posting to a dying pod.
        if self._proxy_proc is not None and self._proxy_proc.poll() is None:
            log.info("killing local skillcacher-proxy (pid=%s)", self._proxy_proc.pid)
            try:
                self._proxy_proc.terminate()
                try:
                    self._proxy_proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._proxy_proc.kill()
            except Exception as e:
                log.warning("local proxy teardown raised: %s", e)
        if self._proxy_log_fh is not None:
            try:
                self._proxy_log_fh.close()
            except Exception:
                pass
            self._proxy_log_fh = None

        # Pull vllm.log BEFORE deleting the pod. The log holds per-request
        # `LMCache hit tokens: N` lines that vllm's response body and /metrics
        # don't expose on the cu12 image.
        if self.log_dump_path is not None and self.pod_id:
            try:
                ssh = await asyncio.to_thread(_get_pod_ssh_info, self.pod_id)
                if ssh is None:
                    log.warning("no SSH info for pod %s — cannot dump vllm.log", self.pod_id)
                else:
                    ssh_host, ssh_port = ssh
                    await asyncio.to_thread(
                        _ssh_dump_vllm_log, ssh_host, ssh_port, self.log_dump_path
                    )
            except Exception as e:
                log.warning("vllm.log dump raised: %s", e)

        log.info("tearing down condition=%s (DELETE %s)", self.condition.name, self.pod_name)
        # Best-effort — logged but not raised, since we're already exiting
        # (or unwinding from a raise inside __aenter__).
        try:
            await asyncio.to_thread(_delete_pod_by_name, self.pod_name)
        except Exception as e:
            log.warning("DELETE-by-name raised: %s", e)
        # Drop the proxy_proc reference once we've tried to kill it; second
        # _cleanup call (defensive) becomes a no-op.
        self._proxy_proc = None

    async def __aexit__(self, exc_type, exc, tb):
        await self._cleanup()
        return False  # do not suppress exceptions
