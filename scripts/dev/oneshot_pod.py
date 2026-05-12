"""Create a single self-bootstrapping vLLM+lmcache pod for dev/validation.

Idempotent: if a pod with $POD_NAME already exists, resume; otherwise create.
Waits for /health on the public RunPod proxy URL, then prints:
  POD_ID=<pod_id>
  PROXY_URL=https://<pod_id>-8000.proxy.runpod.net

The dockerStartCmd is lifted from scripts/dev/race_pod_images.py — proven
working for the T4 smoke (Qwen3-8B + lmcache 0.4.4 + prefix_cache).
Differences from scripts/deploy/_provision.py: this script BAKES bootstrap
into dockerStartCmd, so the pod self-starts vllm without manual SSH.
That avoids the foundation-followup #3 deadlock where wait_running polls
the public URL but bootstrap.sh isn't auto-run on IMAGE-override pods.

Reads from .env: RUNPOD_API_KEY, TAILSCALE_AUTH_KEY, HF_TOKEN.
Optional env: POD_NAME, IMAGE, MODEL_NAME, GPU_TYPE_ID, CLOUD_TYPE,
              VOLUME_ID, WAIT_TIMEOUT_S.

⚠️  SSH-debugging the resulting pod: do NOT use `pkill -f vllm` (or any
`pkill -f ...` that matches a substring of the dockerStartCmd). The
container's PID 1 is `bash -c '<entire dockerStartCmd as one string>'`,
which contains "vllm serve" as a substring. `pkill -f vllm` matches PID 1
and kills the container. Use `pkill -x vllm` (exact name match) or
`pkill <PID>` (specific PID from `ps -eo pid,etime,comm | grep vllm`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API_BASE = "https://rest.runpod.io/v1"


def _load_env() -> None:
    env_file = Path(__file__).resolve().parents[2] / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _http(method: str, path: str, api_key: str, body: dict | None = None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(
        f"{API_BASE}{path}",
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            raw = r.read()
            return json.loads(raw) if raw else None
    except urllib.error.HTTPError as e:
        sys.stderr.write(
            f"[oneshot] HTTP {e.code} {method} {path}: {e.read().decode(errors='replace')}\n"
        )
        sys.exit(1)


def _find_existing(api_key: str, pod_name: str) -> str | None:
    qs = urllib.parse.urlencode({"name": pod_name})
    data = _http("GET", f"/pods?{qs}", api_key)
    pods = data if isinstance(data, list) else (data or {}).get("pods", [])
    for p in pods:
        if p.get("name") == pod_name:
            return p["id"]
    return None


def _resume(api_key: str, pod_id: str) -> None:
    _http("POST", f"/pods/{pod_id}/start", api_key, body={})


def _get_status(api_key: str, pod_id: str) -> dict:
    return _http("GET", f"/pods/{pod_id}", api_key) or {}


def _condition_envs(condition: str) -> str:
    """LMCache env exports per condition. Mirrors scripts/deploy/bootstrap.sh."""
    # Optional debug verbosity for retrieval-side tracing — propagates from
    # host env so a single bench invocation with LMCACHE_LOG_LEVEL=DEBUG
    # surfaces vllm_v1_adapter.py / blender.py decision points in vllm.log.
    log_level = os.environ.get("LMCACHE_LOG_LEVEL", "").strip()
    log_export = f"export LMCACHE_LOG_LEVEL={log_level}\n" if log_level else ""
    if condition == "no_cache":
        return (
            f"{log_export}"
            "export LMCACHE_USE_EXPERIMENTAL=False\n"
            "export LMCACHE_LOCAL_CPU=False\n"
        )
    if condition == "prefix_cache":
        return (
            f"{log_export}"
            "export VLLM_KV_CACHE_TYPE=lmcache\n"
            "export LMCACHE_USE_EXPERIMENTAL=False\n"
            "export LMCACHE_LOCAL_CPU=True\n"
            "export LMCACHE_MAX_LOCAL_CPU_SIZE=40\n"
            "export LMCACHE_CHUNK_SIZE=256\n"
        )
    if condition == "cacheblend":
        # CacheBlend requires layerwise execution + blend-specific configs
        # per LMCache docs (kv_cache_optimizations/blending.html). Without
        # LMCACHE_USE_LAYERWISE=True, vllm crashes at startup; the others
        # are required for the blending compute path.
        # Two values are tunable via host env for T35.5/path-C investigation
        # of the lmcache 0.4.2 blender tensor-dim crash on aggressive
        # permutations (see benchmark/results/audit/cacheblend_blender_dim_mismatch.md).
        chunk_size = os.environ.get("LMCACHE_CHUNK_SIZE_OVERRIDE", "256")
        blend_recompute = os.environ.get("LMCACHE_BLEND_RECOMPUTE_RATIOS_OVERRIDE", "0.15")
        # PYTHONHASHSEED=0 — required for chunk-hash consistency across the
        # multiple TokenDatabase inits the lmcache connector performs (one
        # per KVConnectorRole: WORKER + SCHEDULER). Without it, vllm's
        # `init_none_hash` calls `os.urandom(32)` on every invocation and
        # the per-role NONE_HASH values diverge → store-time and lookup-time
        # chunk hashes never match → 0% retrieval. # Llama-3.3-70B path-C v4_debug; lmcache's own builtin-hash WARNING
        # at token_database.py:143 explicitly calls this out. See
        # benchmark/results/audit/llama_cacheblend_zero_outcome.md.
        return (
            f"{log_export}"
            "export PYTHONHASHSEED=0\n"
            "export VLLM_KV_CACHE_TYPE=lmcache\n"
            "export LMCACHE_USE_EXPERIMENTAL=True\n"
            "export LMCACHE_LOCAL_CPU=True\n"
            "export LMCACHE_MAX_LOCAL_CPU_SIZE=40\n"
            f"export LMCACHE_CHUNK_SIZE={chunk_size}\n"
            "export LMCACHE_ENABLE_BLENDING=True\n"
            "export LMCACHE_USE_LAYERWISE=True\n"
            'export LMCACHE_BLEND_SPECIAL_STR=" # # "\n'
            "export LMCACHE_BLEND_CHECK_LAYERS=1\n"
            f"export LMCACHE_BLEND_RECOMPUTE_RATIOS={blend_recompute}\n"
        )
    raise ValueError(f"unknown CONDITION: {condition}")


def _kv_transfer_flag(condition: str) -> str:
    """Skip --kv-transfer-config for no_cache (vllm runs without LMCache)."""
    if condition == "no_cache":
        return ""
    return (
        '  --kv-transfer-config '
        "'{\"kv_connector\":\"LMCacheConnectorV1\",\"kv_role\":\"kv_both\"}' \\\n"
    )


# LMCache PR #2946 commit (fix/cacheblend-vllm-v0.17.1-compat). We can't `pip
# install` the whole thing on the lightweight image because lmcache's C++
# extensions need cusparse.h (CUDA dev toolkit), which isn't in the runtime
# image. Workaround: curl-overwrite the 5 .py files that PR #2946 changes for
# V1-engine compat, then apply 2 surgical in-place patches. Verified working
# on lmcache/vllm-openai:v0.4.3-lightweight (vllm 0.19.0 + lmcache 0.4.2);
# see benchmark/results/audit/mtrag_sanity_outcome.md for the full saga.
_LMCACHE_PR2946_COMMIT = "9f8aa4d6ee70a2a05657470f3b84d3298c05d8a1"

# Files from PR #2946 that we overwrite wholesale. integration/vllm/vllm_v1_adapter.py
# is excluded — the PR's version imports a sibling `vllm_service_factory` module
# that doesn't exist in lmcache 0.4.2. We patch its single needed line surgically.
_LMCACHE_PATCH_FILES = (
    "v1/compute/attention/flash_attn.py",
    "v1/compute/attention/flash_infer_sparse.py",
    "v1/compute/attention/utils.py",
    "v1/compute/models/base.py",
    "v1/compute/models/utils.py",
)


def _cacheblend_patches() -> str:
    """Four sets of patches required for `vllm serve` + cacheblend (since the
    maintainers ship cacheblend as an offline `blend.py` script and "not planned"
    for vllm serve per LMCache#1936). All idempotent; re-running on a warm pod
    is a no-op.

    1. Overwrite 5 lmcache .py files with PR #2946 versions (vllm V1 engine
       compat: vllm.attention import path, embed_input_ids, lazy flashinfer,
       rope_scaling getattr).
    2. In-place patch lmcache/integration/vllm/vllm_v1_adapter.py to wrap
       `request.all_token_ids` in `list(...)` (msgpack can't encode ConstantList).
    3. In-place patch vllm/v1/worker/gpu_worker.py to call
       `VLLMModelTracker.register_model(ENGINE_NAME, unwrapped_model)` before
       `ensure_kv_transfer_initialized`. Walks `.unwrap()` to peel off
       CUDAGraphWrapper/UBatchWrapper so lmcache sees the real model class.
    4. In-place patch lmcache/v1/compute/blend/blender.py to align k and old_k
       when retrieve_layer's mask-aware skip leaves old_k shorter than
       compute_layer's mask-blind k. Fixes the dim-mismatch crash in
       process_qkv (upstream LMCache issues #1875 / #854 / #938 family).
       Repro + correctness analysis:
       benchmark/results/audit/cacheblend_blender_dim_mismatch.md
    """
    files_curl = " ".join(_LMCACHE_PATCH_FILES)
    return rf"""
echo '[oneshot] cacheblend: overwriting 5 lmcache files from PR #2946'
DST=/usr/local/lib/python3.12/dist-packages/lmcache
RAW=https://raw.githubusercontent.com/LMCache/LMCache/{_LMCACHE_PR2946_COMMIT}
for path in {files_curl}; do
    curl -fsS "$RAW/lmcache/$path" -o "$DST/$path.new" \
      && mv "$DST/$path.new" "$DST/$path" \
      || (echo "[oneshot] FATAL: failed to fetch $path" && sleep infinity)
done

echo '[oneshot] cacheblend: in-place patches (gpu_worker.py + vllm_v1_adapter.py)'
python3 - <<'PYEOF' || (echo '[oneshot] FATAL: in-place patches failed' && sleep infinity)
import pathlib, re, sys

# Patch 1: gpu_worker.py — register unwrapped model before KV connector init.
p = pathlib.Path('/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu_worker.py')
src = p.read_text()
sentinel = '# LMCACHE_CACHEBLEND_PATCH'
if sentinel in src:
    print('[patch] gpu_worker already patched')
else:
    needle = 'ensure_kv_transfer_initialized(self.vllm_config, kv_cache_config)'
    m = re.search(r'^(\s+)' + re.escape(needle), src, re.M)
    if not m:
        print('[patch] FATAL: gpu_worker needle not found'); sys.exit(1)
    indent = m.group(1)
    block = [
        sentinel,
        'try:',
        '    from lmcache.v1.compute.models.utils import VLLMModelTracker',
        '    from lmcache.integration.vllm.utils import ENGINE_NAME',
        '    _lmc_model = self.model_runner.model',
        '    while hasattr(_lmc_model, "unwrap") and callable(getattr(_lmc_model, "unwrap")):',
        '        _lmc_model = _lmc_model.unwrap()',
        '    VLLMModelTracker.register_model(ENGINE_NAME, _lmc_model)',
        'except ImportError:',
        '    pass',
    ]
    patch = ''.join(indent + ln + chr(10) for ln in block)
    p.write_text(src.replace(indent + needle, patch + indent + needle, 1))
    print(f'[patch] gpu_worker register_model inserted at indent={{len(indent)}}')

# Patch 2: vllm_v1_adapter.py — wrap request.all_token_ids in list() so
# msgpack can encode it (vllm 0.19 wraps token IDs in immutable ConstantList).
p2 = pathlib.Path('/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/vllm_v1_adapter.py')
src2 = p2.read_text()
if 'LMCACHE_CACHEBLEND_PATCH' in src2:
    print('[patch] adapter already patched')
else:
    old = '            # all token ids covers the preemption case\n            token_ids = request.all_token_ids\n'
    new = '            # all token ids covers the preemption case\n            # LMCACHE_CACHEBLEND_PATCH: ConstantList -> list for msgpack\n            token_ids = list(request.all_token_ids)\n'
    if old not in src2:
        print('[patch] FATAL: adapter needle not found'); sys.exit(1)
    p2.write_text(src2.replace(old, new, 1))
    print('[patch] adapter all_token_ids wrapped in list()')

# Patch 3: blender.py — fix dim-mismatch crash in process_qkv when
# retrieve_layer's mask-aware skip leaves old_k shorter than compute_layer's
# mask-blind k. The trailing old_k.shape[0] rows of k align with old_k's
# positions. We splice ONLY inside the check_layer block (slicing at the
# top would break attn_metadata alignment in non-check layers and trigger
# CUDA OOB). imp_indices stay in old_k's local space [0, old_k.shape[0])
# for the writeback at later layers; q/k/v/residual/attn_metadata use
# global (full-k space) indices = local + offset. (LMCache #1875/#854/#938.)
p3 = pathlib.Path('/usr/local/lib/python3.12/dist-packages/lmcache/v1/compute/blend/blender.py')
src3 = p3.read_text()
if 'LMCACHE_CACHEBLEND_DIMFIX' in src3:
    print('[patch] blender already dim-fix patched')
else:
    OLD_CHECK = (
        '        if layer_id in self.common_metadata.check_layers:\n'
        '            diff_k = torch.sum(\n'
        '                (k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]\n'
        '            )\n'
        '            total_len = diff_k.shape[0]\n'
        '\n'
        '            assert self.common_metadata.recomp_ratios is not None\n'
        '\n'
        '            # TODO(Jiayi): remove `[0]` hardcode\n'
        '            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])\n'
        '            topk_num = max(topk_num, 1)\n'
        '\n'
        '            top_indices = torch.topk(diff_k, k=topk_num).indices\n'
        '            top_indices, _ = torch.sort(top_indices)\n'
        '\n'
        '            k, v = k[top_indices], v[top_indices]\n'
        '            q = q[top_indices]\n'
        '            residual = residual[top_indices]\n'
        '\n'
        '            logger.debug(f"Number of indices picked: {{len(top_indices)}}")\n'
        '\n'
        '            self.metadata.imp_indices = top_indices\n'
        '            self.metadata.positions = self.metadata.positions[top_indices]\n'
        '            attn_output = attn_output[:topk_num]\n'
        '\n'
        '            attn_metadata.update_from_top_indices(top_indices)\n'
    )
    NEW_CHECK = (
        '        if layer_id in self.common_metadata.check_layers:\n'
        '            # LMCACHE_CACHEBLEND_DIMFIX: align k with old_k for diff_k. The\n'
        '            # trailing old_k.shape[0] rows of k correspond to old_k positions.\n'
        '            # imp_indices stay in old_k local space; q/k/v/residual/attn_metadata\n'
        '            # use global (full-k space) indices = local + offset.\n'
        '            # See LMCache #1875 / #854 / #938.\n'
        '            if k.shape[0] > old_k.shape[0]:\n'
        '                _offset = k.shape[0] - old_k.shape[0]\n'
        '                _k_for_diff = k[_offset:]\n'
        '            else:\n'
        '                _offset = 0\n'
        '                _k_for_diff = k\n'
        '\n'
        '            diff_k = torch.sum(\n'
        '                (_k_for_diff.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1]\n'
        '            )\n'
        '            total_len = diff_k.shape[0]\n'
        '\n'
        '            assert self.common_metadata.recomp_ratios is not None\n'
        '\n'
        '            # TODO(Jiayi): remove `[0]` hardcode\n'
        '            topk_num = int(total_len * self.common_metadata.recomp_ratios[0])\n'
        '            topk_num = max(topk_num, 1)\n'
        '\n'
        '            top_indices = torch.topk(diff_k, k=topk_num).indices\n'
        '            top_indices, _ = torch.sort(top_indices)\n'
        '            _global_top_indices = top_indices + _offset\n'
        '\n'
        '            k, v = k[_global_top_indices], v[_global_top_indices]\n'
        '            q = q[_global_top_indices]\n'
        '            residual = residual[_global_top_indices]\n'
        '\n'
        '            logger.debug(f"Number of indices picked: {{len(top_indices)}}")\n'
        '\n'
        '            self.metadata.imp_indices = top_indices\n'
        '            self.metadata.positions = self.metadata.positions[_global_top_indices]\n'
        '            attn_output = attn_output[:topk_num]\n'
        '\n'
        '            attn_metadata.update_from_top_indices(_global_top_indices)\n'
    )
    if OLD_CHECK not in src3:
        print('[patch] FATAL: blender check_layer block not found'); sys.exit(1)
    p3.write_text(src3.replace(OLD_CHECK, NEW_CHECK, 1))
    print('[patch] blender check_layer block rewritten with global-index mapping')

# Patch 4: token_database.py — fix SegmentTokenDatabase.sep_tokens for tokenizers
# that prepend BOS to encode(). The upstream `encode(special_str)[1:]` is meant
# to strip a leading BOS, but the resulting tokens are the *start-of-string*
# tokenization which differs from the *mid-text* BPE merges by surrounding-space
# context. On Llama-3 specifically, encode("# #") = [128000, 2, 674] →
# [1:] = [2, 674], but mid-text "# #" tokenizes as [674, 674]. The wrong marker
# never matches → 0 segments detected → cacheblend hit rate = 0%. Qwen3 evades
# the bug because its tokenizer doesn't add BOS by default. Fix: encode with a
# leading space and add_special_tokens=False to get the in-text token pattern.
# See benchmark/results/audit/llama_cacheblend_zero_outcome.md.
p4 = pathlib.Path('/usr/local/lib/python3.12/dist-packages/lmcache/v1/token_database.py')
src4 = p4.read_text()
if 'LMCACHE_CACHEBLEND_SEPTOKFIX' in src4:
    print('[patch] token_database already sep-tok-fix patched')
else:
    OLD_SEP = '        self.sep_tokens = self.tokenizer.encode(config.blend_special_str)[1:]\n'
    NEW_SEP = (
        '        # LMCACHE_CACHEBLEND_SEPTOKFIX: encode with leading space + add_special_tokens=False\n'
        '        # so BPE merges match the in-text token pattern. The upstream `encode(s)[1:]` strips\n'
        '        # BOS but leaves the start-of-string sequence (e.g., [2, 674] for Llama-3) that never\n'
        '        # appears mid-text, causing 0% segment detection on any tokenizer that adds BOS.\n'
        '        self.sep_tokens = self.tokenizer.encode(" " + config.blend_special_str, add_special_tokens=False)\n'
    )
    if OLD_SEP not in src4:
        print('[patch] FATAL: token_database sep_tokens line not found'); sys.exit(1)
    p4.write_text(src4.replace(OLD_SEP, NEW_SEP, 1))
    print('[patch] token_database sep_tokens rewritten to use leading-space encode')

# Patch 6 (the harness fix — discovered via a DIAG run):
# vllm gives the STORE path `request.token_ids` (chunk-size-aligned subset)
# but the LOOKUP path `request.all_token_ids` (full prompt). Result: store
# hashes tokens[0:256] but lookup hashes tokens[0:289] → never match → 0%
# hits across all 48 requests in the v4/v5 spikes. Fix: truncate LOOKUP
# tokens to chunk-size alignment so store and lookup hash the same range.
# Short prompts (< chunk_size) get aligned to 0 → lookup_client returns 0
# early which is the same behavior store has on those (mask-skip).
p6 = pathlib.Path('/usr/local/lib/python3.12/dist-packages/lmcache/integration/vllm/vllm_v1_adapter.py')
src6 = p6.read_text()
if 'CACHEBLEND_LOOKUP_ALIGN' in src6:
    print('[patch] vllm_v1_adapter already lookup-align patched')
else:
    OLD6 = (
        '            # all token ids covers the preemption case\n'
        '            # LMCACHE_CACHEBLEND_PATCH: ConstantList -> list for msgpack\n'
        '            token_ids = list(request.all_token_ids)\n'
    )
    NEW6 = (
        '            # all token ids covers the preemption case\n'
        '            # LMCACHE_CACHEBLEND_PATCH: ConstantList -> list for msgpack\n'
        '            token_ids = list(request.all_token_ids)\n'
        '            # CACHEBLEND_LOOKUP_ALIGN: align lookup tokens to chunk_size\n'
        '            # so SegmentTokenDatabase yields the same hash as store\n'
        '            # (which only sees `request.token_ids`, chunk-aligned).\n'
        '            _diag_chunk = self._lmcache_chunk_size\n'
        '            _diag_aligned = (len(token_ids) // _diag_chunk) * _diag_chunk\n'
        '            if _diag_aligned > 0 and _diag_aligned < len(token_ids):\n'
        '                token_ids = token_ids[:_diag_aligned]\n'
    )
    if OLD6 not in src6:
        print('[patch] FATAL: vllm_v1_adapter all_token_ids site not found'); sys.exit(1)
    p6.write_text(src6.replace(OLD6, NEW6, 1))
    print('[patch] vllm_v1_adapter LOOKUP tokens chunk-size aligned')

# Patch 5 (the harness diagnostic — per an external review ):
# Insert print statements at the lookup_client + cache_engine sites that
# identified as the cheapest conclusive trace points to distinguish
# (a) workload-shape mismatch (store keys never appear in lookup keys) from
# (b) per-rank min collapse (one rank returns 0) from (c) batched_contains
# returning 0 despite key match. Gated on SKILLCACHER_LMCACHE_DIAG_PRINTS=true
# so production runs aren't polluted with verbose diag output.
# All double-braces below are escaped because the enclosing
# _cacheblend_patches() returns an rf-string; we want literal single
# braces to survive into the embedded Python source.
import os as _os
if _os.environ.get('SKILLCACHER_LMCACHE_DIAG_PRINTS', '').lower() in ('true', '1', 'yes'):
    # 5a: LMCacheLookupClient.lookup() — print per-rank results after
    # send_and_recv_all returns.
    p5 = pathlib.Path('/usr/local/lib/python3.12/dist-packages/lmcache/v1/lookup_client/lmcache_lookup_client.py')
    src5 = p5.read_text()
    if 'CBLOOKUP_CLIENT_DIAG' in src5:
        print('[patch] lookup_client already diag-patched')
    else:
        OLD5 = '        results = [int.from_bytes(resp, "big") for resp in responses]\n'
        NEW5 = (
            '        results = [int.from_bytes(resp, "big") for resp in responses]\n'
            '        # CBLOOKUP_CLIENT_DIAG\n'
            '        import os as _diag_os\n'
            '        _diag_n = (len(token_ids) if token_ids is not None else 0)\n'
            '        print(f"[CBLOOKUP_CLIENT pid={{_diag_os.getpid()}} lookup_id={{lookup_id}} '
            'ntok={{_diag_n}} blending={{self.enable_blending}} results={{results}} '
            'min={{min(results) if results else 0}}]", flush=True)\n'
        )
        if OLD5 not in src5:
            print('[patch] FATAL: lookup_client send_and_recv_all line not found'); sys.exit(1)
        p5.write_text(src5.replace(OLD5, NEW5, 1))
        print('[patch] lookup_client diag prints inserted')

        # 5b: LMCacheLookupServer.process_request() — print BEFORE/AFTER
        # the lmcache_engine.lookup(...) call so we can correlate
        # client-observed `results` with worker-side hit counts.
        src5b = p5.read_text()
        # BEFORE the if not self.enable_blending: branch
        OLD5B_PIVOT = '                    if not self.enable_blending:\n'
        NEW5B_PIVOT = (
            '                    # CBLOOKUP_SERVER_BEFORE\n'
            '                    import os as _diag_os2\n'
            '                    _diag_meta = self.lmcache_engine.metadata\n'
            '                    print(f"[CBLOOKUP_SERVER_BEFORE pid={{_diag_os2.getpid()}} "\n'
            '                          f"worker={{_diag_meta.worker_id}} lookup_id={{lookup_id}} "\n'
            '                          f"blending={{self.enable_blending}} frames={{len(data_frames)}}]",\n'
            '                          flush=True)\n'
            '                    if not self.enable_blending:\n'
        )
        if OLD5B_PIVOT not in src5b:
            print('[patch] FATAL: server process_request pivot not found'); sys.exit(1)
        src5b = src5b.replace(OLD5B_PIVOT, NEW5B_PIVOT, 1)
        # AFTER both lookup_result assignments — single common point is
        # right before `response = lookup_result.to_bytes(4, "big")`.
        OLD5B_AFTER = '                    response = lookup_result.to_bytes(4, "big")\n'
        NEW5B_AFTER = (
            '                    # CBLOOKUP_SERVER_AFTER\n'
            '                    print(f"[CBLOOKUP_SERVER_AFTER pid={{_diag_os2.getpid()}} "\n'
            '                          f"worker={{_diag_meta.worker_id}} lookup_id={{lookup_id}} "\n'
            '                          f"result={{lookup_result}}]", flush=True)\n'
            '                    response = lookup_result.to_bytes(4, "big")\n'
        )
        if OLD5B_AFTER not in src5b:
            print('[patch] FATAL: server response.to_bytes line not found'); sys.exit(1)
        src5b = src5b.replace(OLD5B_AFTER, NEW5B_AFTER, 1)
        p5.write_text(src5b)
        print('[patch] lookup_server BEFORE/AFTER diag prints inserted')

    # 5c: cache_engine.py — print [CBKEY phase=store ...] in store_layer()
    # and [CBKEY phase=lookup ...] in lookup() use_layerwise branch.
    p5c = pathlib.Path('/usr/local/lib/python3.12/dist-packages/lmcache/v1/cache_engine.py')
    src5c = p5c.read_text()
    if 'CBKEY_DIAG' in src5c:
        print('[patch] cache_engine already diag-patched')
    else:
        OLD5C_STORE = (
            '        for start, end, key in self.token_database.process_tokens(\n'
            '            tokens=tokens, mask=mask, request_configs=request_configs\n'
            '        ):\n'
            '            assert isinstance(key, CacheEngineKey)\n'
            '\n'
            '            keys_multi_layer = key.split_layers(self.num_layers)\n'
            '            # Only check the first layer\n'
            '            if self.storage_manager.contains(\n'
        )
        NEW5C_STORE = (
            '        for start, end, key in self.token_database.process_tokens(\n'
            '            tokens=tokens, mask=mask, request_configs=request_configs\n'
            '        ):\n'
            '            assert isinstance(key, CacheEngineKey)\n'
            '            # CBKEY_DIAG (store_layer)\n'
            '            import os as _diag_os3\n'
            '            print(f"[CBKEY phase=store pid={{_diag_os3.getpid()}} "\n'
            '                  f"worker={{self.metadata.worker_id}} req={{req_id}} "\n'
            '                  f"start={{start}} end={{end}} key={{key}}]", flush=True)\n'
            '\n'
            '            keys_multi_layer = key.split_layers(self.num_layers)\n'
            '            # Only check the first layer\n'
            '            if self.storage_manager.contains(\n'
        )
        if OLD5C_STORE not in src5c:
            print('[patch] FATAL: store_layer process_tokens block not found'); sys.exit(1)
        src5c = src5c.replace(OLD5C_STORE, NEW5C_STORE, 1)

        # lookup site: line 1121 — for loop in use_layerwise branch.
        OLD5C_LOOKUP = (
            '            # TODO: support batched_contains when layerwise is enabled\n'
            '            if self.use_layerwise:\n'
            '                for start, end, key in chunk_info_iterator:\n'
            '                    assert isinstance(key, CacheEngineKey)\n'
        )
        NEW5C_LOOKUP = (
            '            # TODO: support batched_contains when layerwise is enabled\n'
            '            if self.use_layerwise:\n'
            '                for start, end, key in chunk_info_iterator:\n'
            '                    assert isinstance(key, CacheEngineKey)\n'
            '                    # CBKEY_DIAG (lookup)\n'
            '                    import os as _diag_os4\n'
            '                    print(f"[CBKEY phase=lookup pid={{_diag_os4.getpid()}} "\n'
            '                          f"worker={{self.metadata.worker_id}} lookup_id={{lookup_id}} "\n'
            '                          f"start={{start}} end={{end}} key={{key}}]", flush=True)\n'
        )
        if OLD5C_LOOKUP not in src5c:
            print('[patch] FATAL: lookup use_layerwise block not found'); sys.exit(1)
        src5c = src5c.replace(OLD5C_LOOKUP, NEW5C_LOOKUP, 1)
        p5c.write_text(src5c)
        print('[patch] cache_engine CBKEY diag prints inserted (store_layer + lookup)')
else:
    print('[patch] §1.6 diag prints SKIPPED (set SKILLCACHER_LMCACHE_DIAG_PRINTS=true to enable)')
PYEOF
""".strip()


def _render_start_cmd(
    model_name: str,
    model_dir: str,
    needs_lmcache: bool,
    max_model_len: int,
    condition: str,
    *,
    dtype: str,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    expected_lmcache_version: str | None,
    tensor_parallel_size: int = 1,
) -> str:
    """Bash that vllm-openai pods run via `bash -c`. Lifted from
    scripts/dev/race_pod_images.py:render_docker_start_cmd. Sequenced so a
    failure surfaces but the container stays alive (sleep infinity), so we
    can SSH in to debug if something goes wrong."""
    install_lmcache = (
        "pip install --quiet --no-cache-dir lmcache==0.4.4"
        if needs_lmcache
        else "echo 'lmcache pre-baked'"
    )
    # Cacheblend extras: `vllm serve` cacheblend isn't supported upstream
    # ("not planned" per LMCache#1936). Maintainers ship the working setup as
    # an offline `examples/blend_kv_v1/blend.py` script that requires (a)
    # several lmcache V1-engine compat fixes from PR LMCache#2946 and (b) a
    # vllm-side gpu_worker.py patch that calls `VLLMModelTracker.register_model`
    # before KV connector init. Apply both at boot when CONDITION=cacheblend so
    # the same pre-baked image (lmcache/vllm-openai:v0.4.3-lightweight) keeps
    # working for prefix_cache while we get a working cacheblend out of vllm
    # serve. See benchmark/results/audit/mtrag_sanity_outcome.md.
    cacheblend_patches = _cacheblend_patches() if condition == "cacheblend" else "echo '[oneshot] cacheblend patches: skipped (condition!=cacheblend)'"
    # the harness diag-prints flag: propagate from host into the pod's
    # bash environment so the patch python heredoc's
    # os.environ.get('SKILLCACHER_LMCACHE_DIAG_PRINTS') reads it on the pod.
    diag_prints = os.environ.get("SKILLCACHER_LMCACHE_DIAG_PRINTS", "").lower() in ("true", "1", "yes")
    diag_export = "export SKILLCACHER_LMCACHE_DIAG_PRINTS=true\n" if diag_prints else ""
    condition_envs = _condition_envs(condition)
    kv_flag = _kv_transfer_flag(condition)
    # Lmcache version verify: warn-only. We can't pip-install on the cu12
    # lightweight image (cusparse.h missing — see mtrag_sanity_outcome.md),
    # so this is purely a drift guard for the bench JSON's version-record.
    if expected_lmcache_version:
        version_check = (
            f"python3 -c \""
            f"import lmcache; v=lmcache.__version__;"
            f" print('[oneshot] lmcache version:', v);"
            f" assert v == '{expected_lmcache_version}', "
            f"  f'WARN: expected lmcache=={expected_lmcache_version}, got {{v}}'"
            f"\" || echo '[oneshot] WARN: lmcache version drift'"
        )
    else:
        version_check = "echo '[oneshot] no expected lmcache version pinned'"
    # Llama-3.3-70B needs fp8 + max-num-seqs 64 and a higher gpu_memory_utilization
    # than the Qwen3-8B default. Make all three env-knobbable so a single oneshot
    # path serves both. NOTE: vllm 0.19's --dtype only accepts model-dtype values
    # (auto, bfloat16, float16, ...). For fp8 we route DTYPE=fp8 → --dtype auto
    # --quantization fp8 (vllm dynamically quantizes weights at load time).
    if dtype.lower() in ("fp8", "fp4", "awq", "gptq", "int8"):
        dtype_flags = f"--dtype auto --quantization {dtype.lower()}"
    else:
        dtype_flags = f"--dtype {dtype}"
    # Cacheblend interacts badly with vllm's prefix cache: when prefix-cache
    # covers the leading tokens, lmcache's blender gets only a trailing slice
    # and `retrieve_layer` falls into its empty-keys branch (yields None per
    # layer without ever loading buffer_mapping[0]). The first compute_layer(0)
    # then crashes with `Layer 0 is not loaded into GPU buffer.` Disabling
    # vllm's prefix cache for cacheblend keeps lmcache as the sole cache layer.
    # 
    prefix_cache_flag = "--no-enable-prefix-caching" if condition == "cacheblend" else ""
    # the harness finding ( a DIAG run): vllm's chunked_prefill
    # truncates `request.token_ids` to chunk-size multiples for the STORE
    # path while LOOKUP uses `request.all_token_ids` (full prompt). Result:
    # store hashes tokens[0:256] but lookup hashes tokens[0:289] — never match.
    # Disabling chunked_prefill for cacheblend forces vllm to prefill the full
    # prompt in one step, so STORE and LOOKUP see the same token range.
    # Knob: SKILLCACHER_DISABLE_CHUNKED_PREFILL=true (default false, since
    # disabling chunked_prefill costs throughput on long prompts).
    chunked_prefill_flag = (
        "--no-enable-chunked-prefill"
        if condition == "cacheblend" and os.environ.get(
            "SKILLCACHER_DISABLE_CHUNKED_PREFILL", "").lower() in ("true", "1", "yes")
        else ""
    )
    tp_flag = f"--tensor-parallel-size {tensor_parallel_size}" if tensor_parallel_size > 1 else ""
    return f"""
exec > >(tee -a /var/log/oneshot_boot.log) 2>&1
echo "[oneshot] boot recipe started at $(date -Iseconds)"
set -x
mkdir -p /root/.ssh
echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys 2>/dev/null || true
chmod 700 /root/.ssh && chmod 600 /root/.ssh/authorized_keys 2>/dev/null || true
if ! command -v sshd >/dev/null 2>&1 && ! [ -x /usr/sbin/sshd ]; then
  (apt-get update -qq && apt-get install -y -qq openssh-server) >/var/log/apt.log 2>&1 || true
fi
mkdir -p /run/sshd
(service ssh start 2>/dev/null || /usr/sbin/sshd 2>/dev/null) || true

if ! command -v tailscale >/dev/null; then
  curl -fsSL https://tailscale.com/install.sh | sh
fi
tailscale up --auth-key "$TAILSCALE_AUTH_KEY" --hostname "$POD_NAME" --ssh --accept-routes >/var/log/tailscale.log 2>&1 &
sleep 8

{install_lmcache}

{version_check}

{diag_export}{cacheblend_patches}

mkdir -p {model_dir}
# HF_TOKEN read from os.environ (NOT interpolated into the cmdline) so it
# doesn't show up in `ps aux` / /proc/<pid>/cmdline. Single-tenant pod, so
# remote-leak risk is low, but bad hygiene to publish secrets in argv.
python3 -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('{model_name}', local_dir='{model_dir}', token=os.environ['HF_TOKEN'], allow_patterns=['*.json','*.safetensors','*.txt','*.model'])
" || (echo '[oneshot] model download failed' && sleep infinity)

{condition_envs}
nohup vllm serve {model_dir} \\
  --served-model-name {model_name} \\
  --host 0.0.0.0 \\
  --port 8000 \\
  --max-model-len {max_model_len} \\
  --gpu-memory-utilization {gpu_memory_utilization} \\
  --max-num-seqs {max_num_seqs} \\
  {dtype_flags} \\
  --enable-auto-tool-choice \\
  --tool-call-parser hermes \\
  {prefix_cache_flag} \\
  {chunked_prefill_flag} \\
  {tp_flag} \\
{kv_flag}  >/var/log/vllm.log 2>&1 &

sleep infinity
""".strip()


def _wait_running(api_key: str, pod_id: str, timeout_s: int) -> tuple[str | None, str]:
    """Wait for the pod to be reachable on /health, distinguishing two failure
    modes that look identical at the proxy URL but require different responses:

      * "stuck_provisioning" — RunPod accepted the rental (`desiredStatus=RUNNING`
        is set immediately on POST /pods) but never actually scheduled the
        container. The `machine` field stays `{}` and `runtime` never appears.
        The cloudflare HTTP proxy serves 404 because there's no upstream. This
        is a RunPod-side GPU capacity issue, NOT a boot-recipe failure. Common
        on hot GPU types (H100 80GB HBM3) during peak hours.
      * "boot_failed" — the container started (machine populated, runtime
        present, ports mapped) but vllm crashed during init or the cacheblend
        patches failed. /health 404s because PID 1 is on `sleep infinity`. This
        IS something to debug; logs are fetchable via SSH (see
        _ssh_fetch_pod_logs).

    Returns (proxy_url, "ok") on success; (None, "stuck_provisioning") or
    (None, "boot_failed") on the respective failures.

    PROVISIONING_TIMEOUT_S env caps the time spent waiting for `machine` to
    populate before we declare the queue stuck and exit fast. When unset it
    defaults to the full WAIT_TIMEOUT_S window — the queue-patient default.
    Treats `machine={}` as an expected mid-state of the RunPod scheduler
    rather than a failure to fail-fast on.
    """
    proxy_url = f"https://{pod_id}-8000.proxy.runpod.net"
    # Cloudflare 403's urllib's default User-Agent on RunPod proxy URLs;
    # foundation followup #4 — set an explicit UA so /health actually
    # reaches the container.
    probe_headers = {"User-Agent": "skillcacher-oneshot/0.1"}
    # Default: wait the full WAIT_TIMEOUT_S window for the scheduler to find
    # a host. RunPod's REST create returns RUNNING + machine={} when the pod
    # has been queued but no host has been assigned yet — that's the queue
    # state, not a failure. the harness capture work () originally
    # treated 300s as fail-fast and retried at the orchestrator layer, but
    # capacity-flicker scenarios (esp. 2× H100 — see the harness drought,
    # ) routinely sit in queue >30 min for a free GPU. Default
    # WAIT_TIMEOUT_S bumped to 3600s as a result. Override PROVISIONING_TIMEOUT_S
    # explicitly only when you want a tighter inner deadline.
    _ptimeout = os.environ.get("PROVISIONING_TIMEOUT_S")
    if _ptimeout:
        provisioning_timeout_s = min(int(_ptimeout), timeout_s)
    else:
        provisioning_timeout_s = timeout_s
    started = time.time()
    deadline = started + timeout_s
    provisioning_deadline = started + provisioning_timeout_s

    machine_populated = False
    last_status = ""
    last_health = ""
    while time.time() < deadline:
        d = _get_status(api_key, pod_id)
        status = d.get("desiredStatus") or ""
        machine = d.get("machine") or {}
        runtime = d.get("runtime")
        public_ip = d.get("publicIp") or ""
        port_mappings = d.get("portMappings") or {}
        # RunPod's REST API can report machine={} indefinitely even when the
        # pod is fully provisioned (publicIp populated, portMappings populated,
        # /health proxy reachable). Treat any of those as "container exists at
        # the proxy level" for the fail-fast gate. Fixes the case where vllm
        # spends 5+ minutes loading a 70B-param model behind a routed proxy
        # that returns 404/502 until the model finishes loading; previously
        # we'd fail-fast at PROVISIONING_TIMEOUT_S despite the pod being alive.
        provisioned_signal = bool(machine or runtime or public_ip or port_mappings)
        if not machine_populated and provisioned_signal:
            elapsed = int(time.time() - started)
            sys.stderr.write(
                f"[oneshot] container provisioned at +{elapsed}s "
                f"(machineId={d.get('machineId','?')}, machine_keys={list(machine)}, "
                f"publicIp={public_ip!r}); now waiting for /health\n"
            )
            machine_populated = True
        if status != last_status:
            sys.stderr.write(
                f"[oneshot] desiredStatus={status} provisioned={machine_populated}\n"
            )
            last_status = status

        # Fail-fast on stuck provisioning so the caller doesn't burn the full
        # WAIT_TIMEOUT_S budget on a known issue.
        if not machine_populated and time.time() > provisioning_deadline:
            sys.stderr.write(
                f"[oneshot] STUCK with machine={{}} after {provisioning_timeout_s}s. "
                f"This presentation has TWO known causes that look identical here:\n"
                f"  (a) RunPod GPU capacity stall — scheduler can't find a free host\n"
                f"  (b) CUDA driver mismatch — host assigned but its NVIDIA driver "
                f"is older than the image needs (e.g. lmcache/vllm-openai:v0.4.3-lightweight "
                f"needs CUDA>=12.9). Container fails at OCI/runc with `nvidia-container-cli "
                f"requirement error`. This is visible only in the RunPod web console's "
                f"system logs, NOT the REST API.\n"
                f"Mitigations: ensure ALLOWED_CUDA_VERSIONS is set (default \"13.0,12.9\"), "
                f"try CLOUD_TYPE=COMMUNITY, try GPU_TYPE_ID='NVIDIA GeForce RTX 4090' "
                f"(4090 fallback policy), retry off-peak. Check the RunPod web console "
                f"for pod {pod_id} to see the actual container-start error.\n"
            )
            return (None, "stuck_provisioning")

        # Probe /health whenever the pod is claimed to be RUNNING. We used
        # to gate this on `machine_populated`, but RunPod's REST API can
        # report `machine={}` even after the pod is fully up and serving
        # /health=200 (observed on multiple v0.4.3-lightweight boots in
        # /07). The cost of probing too early is one 5s timeout
        # per cycle — much cheaper than missing a working pod.
        if status == "RUNNING":
            try:
                req = urllib.request.Request(
                    f"{proxy_url}/health", method="GET", headers=probe_headers
                )
                with urllib.request.urlopen(req, timeout=5) as r:
                    if r.status == 200:
                        sys.stderr.write(f"[oneshot] /health OK at {proxy_url}\n")
                        return (proxy_url, "ok")
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
                msg = str(e)[:60]
                if msg != last_health:
                    sys.stderr.write(f"[oneshot] /health not yet up: {msg}\n")
                    last_health = msg
        time.sleep(10)

    # Total timeout exhausted. Distinguish: did we ever leave the provisioning
    # queue? If not, blame RunPod capacity. If yes, blame the boot recipe.
    if not machine_populated:
        sys.stderr.write(
            f"[oneshot] STUCK in RunPod provisioning queue: pod {pod_id} never got "
            f"a container start within {timeout_s}s. RunPod capacity issue.\n"
        )
        return (None, "stuck_provisioning")
    sys.stderr.write(
        f"[oneshot] pod {pod_id} container started but /health never came up in "
        f"{timeout_s}s — boot recipe failure. Logs will be fetched if SSH is reachable.\n"
    )
    return (None, "boot_failed")


def _ssh_fetch_pod_logs(api_key: str, pod_id: str, dest_dir: Path) -> bool:
    """Best-effort SSH-fetch /var/log/oneshot_boot.log and /var/log/vllm.log
    from a pod whose /health probe never came up. Called BEFORE the defensive
    DELETE so we don't lose evidence of why boot failed. Returns True if at
    least one file was successfully fetched.

    Mirrors the pattern in src/skillcacher/bench/conditions.py
    (_get_pod_ssh_info + _ssh_dump_vllm_log) but specialized for the boot-
    timeout path: pulls multiple files, tolerates missing files, never raises.
    """
    d = _get_status(api_key, pod_id)
    host = d.get("publicIp")
    port_raw = (d.get("portMappings") or {}).get("22")
    if not host or not port_raw:
        sys.stderr.write(
            f"[oneshot] no SSH portMapping for pod {pod_id} "
            f"(publicIp={host!r}, port22={port_raw!r}) — skipping log fetch\n"
        )
        return False
    ssh_port = int(port_raw)
    key = os.path.expanduser("~/.ssh/runpod_ed25519")
    if not Path(key).exists():
        # Fall back to id_ed25519 if the runpod-specific one is absent
        alt = os.path.expanduser("~/.ssh/id_ed25519")
        if Path(alt).exists():
            key = alt
        else:
            sys.stderr.write(f"[oneshot] no SSH key at {key} or {alt} — skipping log fetch\n")
            return False

    dest_dir.mkdir(parents=True, exist_ok=True)
    any_ok = False
    for remote in ("/var/log/oneshot_boot.log", "/var/log/vllm.log", "/var/log/tailscale.log"):
        local = dest_dir / Path(remote).name
        cmd = [
            "ssh", "-i", key,
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-o", "IdentitiesOnly=yes",
            "-o", "ConnectTimeout=15",
            "-p", str(ssh_port), f"root@{host}",
            f"cat {remote} 2>/dev/null || true",
        ]
        try:
            with local.open("wb") as f:
                r = subprocess.run(cmd, stdout=f, stderr=subprocess.PIPE, timeout=45)
        except (subprocess.TimeoutExpired, OSError) as e:
            sys.stderr.write(f"[oneshot] SSH fetch {remote} errored: {type(e).__name__}: {e}\n")
            continue
        size = local.stat().st_size if local.exists() else 0
        if r.returncode == 0 and size > 0:
            sys.stderr.write(f"[oneshot] fetched {remote} → {local} ({size}B)\n")
            any_ok = True
        else:
            stderr_tail = r.stderr.decode(errors="replace")[:200] if r.stderr else ""
            sys.stderr.write(
                f"[oneshot] fetch {remote} empty/failed (rc={r.returncode}, size={size}B): {stderr_tail}\n"
            )
            if size == 0:
                local.unlink(missing_ok=True)
    return any_ok


def main() -> None:
    _load_env()
    api_key = os.environ.get("RUNPOD_API_KEY", "").strip()
    if not api_key:
        sys.stderr.write("[oneshot] RUNPOD_API_KEY required\n")
        sys.exit(1)
    if not os.environ.get("TAILSCALE_AUTH_KEY"):
        sys.stderr.write("[oneshot] WARNING: TAILSCALE_AUTH_KEY missing; pod won't join tailnet (RunPod proxy URL will still work)\n")
    if not os.environ.get("HF_TOKEN"):
        sys.stderr.write("[oneshot] WARNING: HF_TOKEN missing; gated models will fail to download\n")

    condition = os.environ.get("CONDITION", "prefix_cache")
    if condition not in ("no_cache", "prefix_cache", "cacheblend"):
        sys.stderr.write(f"[oneshot] FATAL: CONDITION must be no_cache|prefix_cache|cacheblend, got {condition!r}\n")
        sys.exit(1)
    # Default POD_NAME folds in CONDITION so different conditions don't collide.
    pod_name = os.environ.get("POD_NAME", f"skillcacher-dev-{condition}")
    # LMCache's pre-baked cu12 lightweight image ships vllm 0.19.0 + lmcache
    # 0.4.2 with the C++ backend already linked against libcudart.so.12 — so
    # `prefix_cache` and `no_cache` work out of the box, no pip install / no
    # silent python-fallback. `vllm/vllm-openai:latest` ships CUDA 13 and
    # silently falls back to the python backend.
    #
    # NOTE: cacheblend is STILL broken on this image — `LMCBlenderBuilder`
    # calls `VLLMModelTracker.get_model("vllm-instance")` during connector
    # init, but nothing in lmcache 0.4.2 OR vllm 0.19.0 ever calls the
    # corresponding `register_model()`. EngineCore crashes with
    # `ValueError: vllm model for vllm-instance not found.` See
    # mtrag_sanity_outcome.md for the full traceback and resolution paths.
    image = os.environ.get("IMAGE", "lmcache/vllm-openai:v0.4.3-lightweight")
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-8B")
    # H100 80GB HBM3 is the canonical default — required for the Llama-3.3-70B
    # fp8 canonical run. But H100 capacity routinely stalls in RunPod's
    # provisioning queue during peak hours (see _wait_running's
    # "stuck_provisioning" path).
    #
    # 4090 fallback policy (temporary, debug/validation only):
    # When the canonical workload doesn't strictly need H100-class memory or
    # throughput, RTX 4090 (24GB) is an acceptable temporary alternative for
    # CORRECTNESS validation — e.g., reproducing the cacheblend dim-mismatch
    # bug, validating the single-swap workaround on full_5, smoke-testing the
    # bench harness end-to-end on Qwen3-8B. Cost is ~10x cheaper and capacity
    # is generally available.
    #
    # Use 4090 by setting:
    #   GPU_TYPE_ID="NVIDIA GeForce RTX 4090"
    #   MAX_MODEL_LEN<=8192      (24GB doesn't fit Qwen3-8B at 32k context)
    #   MAX_NUM_SEQS<=8
    #
    # Do NOT use 4090 for:
    #   - canonical runs (Llama-3.3-70B fp8 needs 80GB+)
    #   - Hit-rate numbers reported as final canonical results (re-run on H100
    #     so any GPU-arch interaction is held constant with the published
    #     baseline)
    gpu_type_id = os.environ.get("GPU_TYPE_ID", "NVIDIA H100 80GB HBM3")
    cloud_type = os.environ.get("CLOUD_TYPE", "SECURE")
    volume_id = os.environ.get("VOLUME_ID", "").strip() or None
    # 1hr default — queue-patient. Capacity-tight 2× H100 routinely sits in
    # RunPod's provisioning queue for 30+ min before a host frees. The earlier
    # 1200s default fail-fast'd through real recoverable queue waits (the harness
    # 12 consecutive synchronous rejections on
    # COMMUNITY before pivoting to SECURE+queue). Override only if you want a
    # tighter outer deadline (e.g., a debug spike where giving up early is
    # cheaper than burning user-facing latency).
    timeout_s = int(os.environ.get("WAIT_TIMEOUT_S", "3600"))
    # Default sized for Llama-3.3-70B fp8 on H100 80GB after vllm v0.19's CUDA
    # graph memory accounting change (the harness verify, ): at the prior
    # 32K + GMU=0.85 defaults, vllm v0.19 reports `Available KV cache memory:
    # -2.91 GiB` and refuses to start. At GMU=0.92, available KV is +2.63 GiB
    # which only fits ~8608 tokens of context. 8192 is the safe round number.
    # Qwen3-8B (smaller weights) tolerates the older 32K + 0.85 — pass
    # MAX_MODEL_LEN=32768 GPU_MEMORY_UTILIZATION=0.85 to restore those.
    max_model_len = int(os.environ.get("MAX_MODEL_LEN", "8192"))
    dtype = os.environ.get("DTYPE", "auto")
    max_num_seqs = int(os.environ.get("MAX_NUM_SEQS", "64"))
    # GMU bumped from 0.85 to 0.92 for v0.19 compat (see MAX_MODEL_LEN comment).
    gpu_memory_utilization = float(os.environ.get("GPU_MEMORY_UTILIZATION", "0.92"))
    # Multi-GPU knobs for capture/E2 workloads where real CC traffic produces
    # >8K-token prompts (the harness finding: claude -p one-shot sends ~50K
    # tokens of overhead). 2× H100 80GB with TP=2 fits 64K+ context easily;
    # the canonical bench remains 1× by default for the harness reproducibility.
    gpu_count = int(os.environ.get("GPU_COUNT", "1"))
    tensor_parallel_size = int(
        os.environ.get("TENSOR_PARALLEL_SIZE", str(gpu_count))
    )
    # Optional drift guard. We can't pip-install on the cu12 image (cusparse.h
    # missing), so this is warn-only — the real value is recorded in vllm.log
    # at boot for the bench JSON.
    expected_lmcache_version = os.environ.get("EXPECTED_LMCACHE_VERSION", "0.4.2") or None
    # Container disk: 100GB fits Qwen3-8B + cu12 image + LMCache caches with
    # room to spare. Llama-3.3-70B fp8 needs ≥80GB just for weights, so the
    # default is bumped to 150GB; sets `CONTAINER_DISK_GB=200` if it also
    # writes a per-request trace store on the pod.
    container_disk_gb = int(os.environ.get("CONTAINER_DISK_GB", "150"))
    # CUDA driver filter: lmcache/vllm-openai:v0.4.3-lightweight requires CUDA
    # >= 12.9 in the host driver. RunPod's scheduler does NOT filter on host
    # CUDA version unless we pass `allowedCudaVersions` explicitly. Without
    # this, hosts with older drivers (~CUDA 12.4-12.8 still common) fail
    # container start with `nvidia-container-cli: requirement error: cuda>=12.9`
    # at the OCI/runc layer. This presents as `machine={}` + /health 404 —
    # indistinguishable from a true capacity stall via the REST API alone (see
    # benchmark/results/audit/runpod_cuda_driver_mismatch.md / memory entry).
    # Default value matches what v0.4.3-lightweight needs; override via env if
    # using a different image.
    allowed_cuda_csv = os.environ.get("ALLOWED_CUDA_VERSIONS", "13.0,12.9").strip()
    allowed_cuda_versions = [v.strip() for v in allowed_cuda_csv.split(",") if v.strip()] if allowed_cuda_csv else None

    needs_lmcache = "lmcache" not in image
    model_dir = f"/models/{model_name.split('/')[-1].lower()}"

    public_key = os.environ.get("SSH_PUBLIC_KEY", "").strip()
    if not public_key:
        for pub_path in ("~/.ssh/runpod_ed25519.pub", "~/.ssh/id_ed25519.pub"):
            full = os.path.expanduser(pub_path)
            if os.path.exists(full):
                public_key = open(full).read().strip()
                sys.stderr.write(f"[oneshot] using public key from {pub_path}\n")
                break
    if not public_key:
        sys.stderr.write("[oneshot] WARNING: no public key found; SSH access will be unavailable\n")

    sys.stderr.write(f"[oneshot] pod_name={pod_name} image={image} model={model_name} condition={condition}\n")
    existing = _find_existing(api_key, pod_name)
    if existing:
        sys.stderr.write(f"[oneshot] pod exists ({existing}); resuming...\n")
        _resume(api_key, existing)
        pod_id = existing
    else:
        body = {
            "name": pod_name,
            "computeType": "GPU",
            "gpuTypeIds": [gpu_type_id],
            "gpuCount": gpu_count,
            "containerDiskInGb": container_disk_gb,
            "volumeInGb": 0,
            "volumeMountPath": "/models",
            "cloudType": cloud_type,
            "interruptible": False,
            "ports": ["22/tcp", "8000/http"],
            **({"allowedCudaVersions": allowed_cuda_versions} if allowed_cuda_versions else {}),
            "env": {
                "TAILSCALE_AUTH_KEY": os.environ.get("TAILSCALE_AUTH_KEY", ""),
                "HF_TOKEN": os.environ.get("HF_TOKEN", ""),
                "POD_NAME": pod_name,
                "PUBLIC_KEY": public_key,
                "MODEL_NAME": model_name,
            },
            "imageName": image,
            "dockerEntrypoint": ["/bin/bash", "-c"],
            "dockerStartCmd": [_render_start_cmd(
                model_name, model_dir, needs_lmcache, max_model_len, condition,
                dtype=dtype,
                max_num_seqs=max_num_seqs,
                gpu_memory_utilization=gpu_memory_utilization,
                expected_lmcache_version=expected_lmcache_version,
                tensor_parallel_size=tensor_parallel_size,
            )],
        }
        if volume_id:
            body["networkVolumeId"] = volume_id
        sys.stderr.write(f"[oneshot] creating pod...\n")
        data = _http("POST", "/pods", api_key, body=body)
        pod_id = data["id"]
        sys.stderr.write(f"[oneshot] created pod {pod_id}\n")

    sys.stderr.write(f"[oneshot] waiting for /health (timeout={timeout_s}s)...\n")
    proxy_url, reason = _wait_running(api_key, pod_id, timeout_s)
    if reason != "ok":
        # Three exit codes communicate intent to callers:
        #   1 = boot_failed (container started, vllm/cacheblend died — needs debug)
        #   2 = stuck_provisioning (RunPod capacity issue, not our code)
        # Callers (cacheblend_proof.sh, bench harness retry loops) should treat
        # these differently: exit-2 = retry/different GPU, exit-1 = investigate.
        exit_code = 1 if reason == "boot_failed" else 2

        # Only fetch SSH logs when the container actually started — there's
        # nothing to capture when the pod is still in RunPod's queue.
        debug_dir = Path(
            os.environ.get("DEBUG_LOG_DIR", "/tmp/skillcacher_oneshot_debug")
        ) / pod_id
        if reason == "boot_failed":
            sys.stderr.write(f"[oneshot] fetching pod logs to {debug_dir} before DELETE\n")
            try:
                _ssh_fetch_pod_logs(api_key, pod_id, debug_dir)
            except Exception as e:
                sys.stderr.write(f"[oneshot] log fetch errored: {type(e).__name__}: {e}\n")

        # KEEP_POD=1 is the manual-debug knob: skip DELETE so the operator can
        # SSH in and inspect a live failed pod (only useful for boot_failed —
        # for stuck_provisioning there's no container to inspect).
        if os.environ.get("KEEP_POD") == "1":
            sys.stderr.write(
                f"[oneshot] KEEP_POD=1: preserving pod {pod_id} ({reason}). "
                f"{'Logs at ' + str(debug_dir) + '. ' if reason == 'boot_failed' else ''}"
                f"Tear down manually when done.\n"
            )
            sys.exit(exit_code)

        # Defense-in-depth: a created-but-unreachable pod is still billing GPU.
        # Don't trust the caller to clean up — DELETE here regardless of which
        # failure mode.
        sys.stderr.write(f"[oneshot] DELETING pod {pod_id} ({reason})\n")
        try:
            _http("DELETE", f"/pods/{pod_id}", api_key)
        except Exception as e:
            sys.stderr.write(
                f"[oneshot] DELETE failed: {e} — pod {pod_id} may still be billing; "
                f"clean up manually via the RunPod console.\n"
            )
        sys.exit(exit_code)
    print(f"POD_ID={pod_id}")
    print(f"PROXY_URL={proxy_url}")


if __name__ == "__main__":
    main()
