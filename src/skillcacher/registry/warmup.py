"""Pre-seed the cache with all installed skill prefixes at proxy startup.

For each (skill, anchor_bytes) in the prefix index:
  1. Tokenize the rendered prefix with the served-model tokenizer
  2. Register in span_registry
  3. Issue a one-shot prefill request to vLLM containing only those tokens
  4. Pin the resulting KV via Controller.Pin (no TTL = persist)

For ~50 skills × 3 anchor lengths = 150 warm-up prefills, each ~1-4K
tokens. Cold-start cost: ~30-60s on a warm pod."""
from __future__ import annotations

import logging
from typing import Awaitable, Callable

from transformers import AutoTokenizer

from skillcacher.tagging.skill_prefix_index import SkillPrefixIndex
from skillcacher.registry.span_registry import SpanRegistry
from skillcacher.registry.lmcache_controller import LMCacheController

log = logging.getLogger("skillcacher.warmup")

WarmupFn = Callable[[list[int]], Awaitable[None]]


async def _noop_warmup(token_ids: list[int]) -> None:
    """Default warmup is a no-op — the pin call alone is enough when the
    KV is already in cache from a prior bench run; for a fresh cache,
    the caller passes a real warmup_fn that issues a prefill request."""
    return


async def pre_seed_skills(
    index: SkillPrefixIndex,
    registry: SpanRegistry,
    controller: LMCacheController | None,
    tokenizer_name: str,
    *,
    warmup_fn: WarmupFn = _noop_warmup,
) -> int:
    """Walk the prefix index, register each entry, issue a warm-up prefill,
    optionally pin the KV. Returns the number of spans pre-seeded.

    When ``controller`` is None (no lmcache-shim configured), the pin step
    is skipped — the warmup prefill alone populates lmcache via vLLM's
    built-in kv_transfer integration. The KV may then be evicted under
    memory pressure, but for typical pre-seed sizes (~20 skills × 3 anchors
    × ~2K tokens ≈ 120K tokens) eviction is unlikely on a single-tenant
    bench pod."""
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    n = 0
    for entry in index.entries():
        token_ids = tok.encode(entry.rendered_prefix.decode("utf-8", errors="replace"), add_special_tokens=False)
        span_id = f"skill:{entry.skill_id}:anchor_{entry.anchor_bytes}"
        registry.register(span_id, token_ids=token_ids, skill_id=entry.skill_id, anchor_bytes=entry.anchor_bytes)
        try:
            await warmup_fn(token_ids)
        except Exception as e:
            log.warning("warmup_fn failed for %s: %s", span_id, e)
        if controller is not None:
            ok = await controller.pin(token_ids, ttl=None)
            if not ok:
                log.warning("pin failed for %s (continuing)", span_id)
        n += 1
    log.info("pre-seeded %d skill spans", n)
    return n


async def pre_seed_statistical_spans(
    spans_file,
    registry: SpanRegistry,
    controller: LMCacheController | None,
    *,
    warmup_fn: WarmupFn = _noop_warmup,
) -> int:
    """pre-seed statistically-mined spans from a JSONL file.

    Each line in `spans_file` is one record from
    `statistical_miner.mine_spans` serialized as JSON, with at least
    ``fingerprint`` and ``token_ids`` keys. Registers each as a
    ``source="statistical"`` span (skill_id ``_statistical``,
    anchor_bytes ``0``), issues a warmup prefill so the KV lands in
    lmcache the same way structural pre-seeds do, and pins via the
    Controller if one is wired up.

    Returns the number of statistical spans pre-seeded. Zero (no-op) if
    the file is missing or empty — this is the "structural-only" arm of
    the §3 ablation."""
    import json
    from pathlib import Path
    p = Path(spans_file)
    if not p.exists():
        return 0
    n = 0
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line:
            continue
        rec = json.loads(line)
        token_ids = rec["token_ids"]
        fp = rec["fingerprint"]
        span_id = f"stat:{fp}"
        registry.register(
            span_id, token_ids=token_ids,
            skill_id="_statistical", anchor_bytes=0,
            source="statistical",
        )
        try:
            await warmup_fn(token_ids)
        except Exception as e:
            log.warning("warmup_fn failed for %s: %s", span_id, e)
        if controller is not None:
            ok = await controller.pin(token_ids, ttl=None)
            if not ok:
                log.warning("pin failed for %s (continuing)", span_id)
        n += 1
    log.info("pre-seeded %d statistical spans from %s", n, p)
    return n


async def warmup_via_litellm(token_ids: list[int], *, served_model: str, api_base: str, api_key: str) -> None:
    """Concrete warmup_fn: issue a 1-token completion request to vLLM with
    the prefix as prompt; the engine prefills and LMCache stores the KV."""
    import litellm
    # vLLM accepts raw token IDs via the `prompt` field in /v1/completions,
    # but litellm wraps chat. Use a stub user message + max_tokens=1.
    # NB: this is approximate — the chat template will prepend tokens,
    # offsetting the cache by a fixed prefix. For exact alignment we'd use
    # vLLM's /v1/completions directly with the raw tokens.
    from transformers import AutoTokenizer
    from skillcacher.proxy.litellm_bridge import _normalize_api_base
    tok = AutoTokenizer.from_pretrained(served_model)
    text = tok.decode(token_ids, skip_special_tokens=False)
    await litellm.acompletion(
        model=f"hosted_vllm/{served_model}",
        api_base=_normalize_api_base(api_base),
        api_key=api_key or "EMPTY",
        messages=[{"role": "user", "content": text}],
        max_tokens=1,
    )
