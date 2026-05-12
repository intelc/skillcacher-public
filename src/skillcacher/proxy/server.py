"""FastAPI proxy with LiteLLM bridge, span tagger v2, Controller.Lookup,
stdout-tail telemetry, and pre-seed of installed skill bodies."""
from __future__ import annotations

import asyncio
import json
import time
import uuid
import logging

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from skillcacher.settings import Settings
from skillcacher.proxy.litellm_bridge import call_unary, call_streaming
from skillcacher.proxy.trace_store import TraceStore, RequestRecord, ChunkHitMap
from skillcacher.proxy.assemble import (
    assemble_and_tokenize, assemble_and_tokenize_with_index,
)
from skillcacher.proxy.cc_segment_parser import (
    is_enabled as cc_segment_parser_enabled,
    rewrite_request_body as cc_segment_rewrite,
)
from skillcacher.proxy.runpod_logs import tail_for_reqid
from skillcacher.tagging.skill_prefix_index import SkillPrefixIndex
from skillcacher.registry.span_registry import SpanRegistry
from skillcacher.registry.lmcache_controller import LMCacheController
from skillcacher.registry.warmup import (
    pre_seed_skills,
    pre_seed_statistical_spans,
    warmup_via_litellm,
)

log = logging.getLogger("skillcacher.proxy")


def build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    store = TraceStore(settings.trace_dir)
    store.init_schema()

    registry = SpanRegistry(settings.span_registry_path)
    registry.init_schema()
    prefix_index = SkillPrefixIndex(settings.parsed_skill_dirs())
    controller = (
        LMCacheController(settings.lmcache_shim_url, settings.lmcache_shim_api_key)
        if settings.lmcache_shim_url and settings.enable_lookup
        else None
    )

    app = FastAPI(title="skillcacher-proxy", version="0.0.3")

    @app.on_event("startup")
    async def _startup():
        if settings.enable_pre_seed:
            log.info(
                "pre-seeding %d entries from prefix index (controller=%s)",
                len(prefix_index.entries()),
                "shim" if controller is not None else "none — pin skipped",
            )
            async def warmup(tids: list[int]) -> None:
                await warmup_via_litellm(
                    tids,
                    served_model=settings.backend_model,
                    api_base=settings.backend_url,
                    api_key=settings.backend_api_key,
                )
            await pre_seed_skills(prefix_index, registry, controller, settings.tokenizer_name, warmup_fn=warmup)
            # optionally pre-seed statistically-mined spans
            # from a JSONL file. No-op when SKILLCACHER_STATISTICAL_SPANS_FILE
            # is unset/missing — that's the structural-only ablation baseline.
            if settings.statistical_spans_file:
                await pre_seed_statistical_spans(
                    settings.statistical_spans_file,
                    registry, controller, warmup_fn=warmup,
                )

    @app.get("/health")
    async def health():
        return {"ok": True}

    @app.post("/v1/messages")
    async def messages(request: Request):
        body = await request.json()
        # inject ` # # ` separators around recognized CC
        # structural blocks BEFORE forwarding to the backend, so
        # cacheblend's segment detector can find chunks in natural CC
        # traffic. Idempotent + flag-gated; returns body unchanged if it
        # doesn't look CC-shaped.
        if cc_segment_parser_enabled():
            body = cc_segment_rewrite(body)
        if body.get("stream"):
            return await _handle_stream(body, settings, store, registry, prefix_index, controller)
        return await _handle_unary(body, settings, store, registry, prefix_index, controller)

    return app


async def _gather_lookups(controller, registry, skill_ids):
    if controller is None or not skill_ids:
        return []
    spans = registry.spans_referenced_by(skill_ids=skill_ids)
    if not spans:
        return []
    results = await asyncio.gather(*[controller.lookup(s.token_ids) for s in spans])
    out: list[ChunkHitMap] = []
    for span, res in zip(spans, results):
        registry.record_lookup_result(span.span_id, hit_tokens=res.hit_tokens, total_tokens=res.total_tokens)
        out.append(ChunkHitMap(
            span_id=span.span_id,
            chunk_hits=res.chunk_hits,
            total_chunks=res.total_chunks,
            hit_tokens=res.hit_tokens,
            total_tokens=res.total_tokens,
        ))
    return out


def _skill_ids_from_tags_and_index(prefix_index, body) -> list[str]:
    """Quick heuristic: scan all text blocks for known prefix matches and
    return the set of matched skill IDs. Used to scope the Lookup batch."""
    seen: set[str] = set()
    for blk in _iter_text_bytes(body):
        for m in prefix_index.find_matches(blk):
            seen.add(m.skill_id)
    return sorted(seen)


def _iter_text_bytes(body: dict):
    sys_b = body.get("system")
    if isinstance(sys_b, str):
        yield sys_b.encode("utf-8")
    elif isinstance(sys_b, list):
        for b in sys_b:
            if b.get("type") == "text":
                yield b.get("text", "").encode("utf-8")
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            yield c.encode("utf-8")
        elif isinstance(c, list):
            for b in c:
                if b.get("type") == "text":
                    yield b.get("text", "").encode("utf-8")


async def _handle_unary(body, settings, store, registry, prefix_index, controller):
    request_id = "req_" + uuid.uuid4().hex[:16]
    session_id = _derive_session_id(body)
    ts_start = time.time()
    try:
        prompt_tokens, span_tags = assemble_and_tokenize_with_index(body, settings.tokenizer_name, prefix_index)
    except Exception as e:
        log.warning("assemble_failed: %s", e)
        prompt_tokens, span_tags = [], []
    skill_ids = _skill_ids_from_tags_and_index(prefix_index, body)
    lookups = await _gather_lookups(controller, registry, skill_ids)

    try:
        anth_resp, raw = await call_unary(
            body, served_model=settings.backend_model,
            api_base=settings.backend_url, api_key=settings.backend_api_key,
            max_completion_tokens=settings.backend_max_completion_tokens,
        )
    except Exception as e:
        log.exception("backend_call_failed")
        return JSONResponse({"type": "error", "error": {"type": "api_error", "message": str(e)}}, status_code=502)
    ts_end = time.time()

    engine_total = None
    engine_load = None
    if settings.enable_stdout_tail and settings.runpod_api_key and settings.runpod_pod_id:
        try:
            tail = await tail_for_reqid(
                api_key=settings.runpod_api_key, pod_id=settings.runpod_pod_id,
                target_reqid=raw.get("id", ""), timeout_s=30.0, poll_interval_s=1.0,
            )
            if tail:
                engine_total = tail["hit_tokens"]
                engine_load = tail.get("need_to_load")
        except Exception:
            log.exception("stdout_tail_failed")

    try:
        usage = raw.get("usage", {})
        store.write(RequestRecord(
            request_id=request_id, session_id=session_id,
            ts_start=ts_start, ts_end=ts_end,
            prompt_tokens=prompt_tokens, response_tokens=[],
            prompt_token_count=usage.get("prompt_tokens", len(prompt_tokens)),
            response_token_count=usage.get("completion_tokens", 0),
            cache_read_tokens=engine_total or 0,
            cache_recompute_tokens=0,
            ttft_ms=(ts_end - ts_start) * 1000,
            request_body_json=json.dumps(body),
            span_tags=span_tags,
            span_lookups=lookups,
            engine_total_hit_tokens=engine_total,
            engine_load_tokens=engine_load,
        ))
    except Exception:
        log.exception("trace_write_failed")
    return JSONResponse(anth_resp)


async def _handle_stream(body, settings, store, registry, prefix_index, controller):
    request_id = "req_" + uuid.uuid4().hex[:16]
    session_id = _derive_session_id(body)
    ts_start = time.time()
    try:
        prompt_tokens, span_tags = assemble_and_tokenize_with_index(body, settings.tokenizer_name, prefix_index)
    except Exception:
        prompt_tokens, span_tags = [], []
    skill_ids = _skill_ids_from_tags_and_index(prefix_index, body)
    lookups = await _gather_lookups(controller, registry, skill_ids)

    async def gen():
        usage = {"prompt_tokens": 0, "completion_tokens": 0}
        chatcmpl_id = ""
        async for ev in call_streaming(
            body, served_model=settings.backend_model,
            api_base=settings.backend_url, api_key=settings.backend_api_key,
            max_completion_tokens=settings.backend_max_completion_tokens,
        ):
            if ev["type"] == "message_start":
                chatcmpl_id = ev["message"].get("id", "")
            elif ev["type"] == "message_delta":
                usage = {
                    "prompt_tokens": ev["usage"]["input_tokens"],
                    "completion_tokens": ev["usage"]["output_tokens"],
                }
            yield f"data: {json.dumps(ev)}\n\n".encode("utf-8")
        ts_end = time.time()

        engine_total = None
        engine_load = None
        if settings.enable_stdout_tail and settings.runpod_api_key and settings.runpod_pod_id and chatcmpl_id:
            try:
                tail = await tail_for_reqid(
                    api_key=settings.runpod_api_key, pod_id=settings.runpod_pod_id,
                    target_reqid=chatcmpl_id, timeout_s=30.0, poll_interval_s=1.0,
                )
                if tail:
                    engine_total = tail["hit_tokens"]
                    engine_load = tail.get("need_to_load")
            except Exception:
                log.exception("stdout_tail_failed")

        try:
            store.write(RequestRecord(
                request_id=request_id, session_id=session_id,
                ts_start=ts_start, ts_end=ts_end,
                prompt_tokens=prompt_tokens, response_tokens=[],
                prompt_token_count=usage.get("prompt_tokens", len(prompt_tokens)),
                response_token_count=usage.get("completion_tokens", 0),
                cache_read_tokens=engine_total or 0,
                cache_recompute_tokens=0,
                ttft_ms=(ts_end - ts_start) * 1000,
                request_body_json=json.dumps(body),
                span_tags=span_tags,
                span_lookups=lookups,
                engine_total_hit_tokens=engine_total,
                engine_load_tokens=engine_load,
            ))
        except Exception:
            log.exception("trace_write_failed")

    return StreamingResponse(gen(), media_type="text/event-stream")


def _derive_session_id(body: dict) -> str:
    import hashlib
    h = hashlib.sha256()
    sys_b = body.get("system")
    if isinstance(sys_b, str):
        h.update(sys_b.encode("utf-8"))
    elif isinstance(sys_b, list):
        for b in sys_b:
            if b.get("type") == "text":
                h.update(b.get("text", "").encode("utf-8"))
    msgs = body.get("messages") or []
    for m in msgs[:1]:
        c = m.get("content")
        if isinstance(c, str):
            h.update(c.encode("utf-8"))
        elif isinstance(c, list):
            for b in c:
                if b.get("type") == "text":
                    h.update(b.get("text", "").encode("utf-8"))
    return "sess_" + h.hexdigest()[:16]


def main():
    # The proxy is spawned as a subprocess by ConditionLifecycle, which means
    # the parent's logging.basicConfig (set by bench/cli.py) doesn't apply.
    # Without this, log.info("pre-seeding …") and similar info-level lines
    # never reach proxy.log under default Python logging (root logger is
    # WARNING by default). a coarser default fixes that.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    settings = Settings()
    app = build_app(settings)
    uvicorn.run(app, host=settings.proxy_host, port=settings.proxy_port)
