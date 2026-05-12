import asyncio
import json
import os
import socket
import threading
import time
from pathlib import Path
from contextlib import contextmanager

import httpx
import pytest
import uvicorn
from fastapi import FastAPI


FIXTURE = Path(__file__).parent.parent / "fixtures" / "claude_code_simple.json"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _fake_backend_app() -> FastAPI:
    app = FastAPI()

    @app.post("/v1/chat/completions")
    async def chat(req: dict):
        return {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "model": req["model"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "ok"},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        }

    return app


@contextmanager
def _serve(app, port: int):
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error", lifespan="off")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    deadline = 5.0
    while deadline > 0:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.1):
                break
        except OSError:
            time.sleep(0.05)
            deadline -= 0.05
    try:
        yield
    finally:
        server.should_exit = True
        t.join(timeout=2)


@pytest.mark.asyncio
async def test_proxy_roundtrip(tmp_path, monkeypatch):
    backend_port = _free_port()
    proxy_port = _free_port()

    monkeypatch.setenv("SKILLCACHER_BACKEND_URL", f"http://127.0.0.1:{backend_port}")
    monkeypatch.setenv("SKILLCACHER_PROXY_PORT", str(proxy_port))
    monkeypatch.setenv("SKILLCACHER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("SKILLCACHER_TOKENIZER_NAME", "NousResearch/Meta-Llama-3-8B")
    monkeypatch.setenv("SKILLCACHER_SPAN_REGISTRY_PATH", str(tmp_path / "span_registry.sqlite"))
    monkeypatch.setenv("SKILLCACHER_SKILL_DIRS", "")
    monkeypatch.setenv("SKILLCACHER_ENABLE_PRE_SEED", "false")
    monkeypatch.setenv("SKILLCACHER_ENABLE_STDOUT_TAIL", "false")

    from skillcacher.proxy.server import build_app
    backend = _fake_backend_app()
    proxy = build_app()

    with _serve(backend, backend_port), _serve(proxy, proxy_port):
        req = json.loads(FIXTURE.read_text())
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(f"http://127.0.0.1:{proxy_port}/v1/messages", json=req)

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["type"] == "message"
    assert body["role"] == "assistant"
    assert body["content"][0]["text"] == "ok"
    assert body["usage"]["input_tokens"] == 7

    # Trace was written
    assert (tmp_path / "traces.sqlite").exists()


@pytest.mark.asyncio
async def test_proxy_streaming(tmp_path, monkeypatch):
    backend_port = _free_port()
    proxy_port = _free_port()

    monkeypatch.setenv("SKILLCACHER_BACKEND_URL", f"http://127.0.0.1:{backend_port}")
    monkeypatch.setenv("SKILLCACHER_PROXY_PORT", str(proxy_port))
    monkeypatch.setenv("SKILLCACHER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("SKILLCACHER_TOKENIZER_NAME", "NousResearch/Meta-Llama-3-8B")
    monkeypatch.setenv("SKILLCACHER_SPAN_REGISTRY_PATH", str(tmp_path / "span_registry.sqlite"))
    monkeypatch.setenv("SKILLCACHER_SKILL_DIRS", "")
    monkeypatch.setenv("SKILLCACHER_ENABLE_PRE_SEED", "false")
    monkeypatch.setenv("SKILLCACHER_ENABLE_STDOUT_TAIL", "false")

    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat(req: dict):
        from fastapi.responses import StreamingResponse as SR
        async def gen():
            yield 'data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"he"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"content":"llo"},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}\n\n'
            yield 'data: [DONE]\n\n'
        return SR(gen(), media_type="text/event-stream")

    from skillcacher.proxy.server import build_app
    proxy = build_app()

    with _serve(backend, backend_port), _serve(proxy, proxy_port):
        req = json.loads(FIXTURE.read_text())
        req["stream"] = True
        async with httpx.AsyncClient(timeout=10) as client:
            async with client.stream("POST", f"http://127.0.0.1:{proxy_port}/v1/messages", json=req) as r:
                events: list[dict] = []
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload and payload != "[DONE]":
                            events.append(json.loads(payload))

    types = [e.get("type") for e in events]
    assert "message_start" in types
    assert "content_block_start" in types
    assert "content_block_delta" in types
    assert "content_block_stop" in types
    assert "message_stop" in types
    text = "".join(
        e["delta"]["text"] for e in events
        if e.get("type") == "content_block_delta" and e["delta"].get("type") == "text_delta"
    )
    assert text == "hello"
    deltas = [e for e in events if e.get("type") == "message_delta"]
    assert len(deltas) == 1
    assert deltas[0]["usage"]["input_tokens"] == 3
    assert deltas[0]["usage"]["output_tokens"] == 2

    # Confirm the trace was written even though the stream ended successfully
    # (post-yield trace_store.write must have run).
    assert (tmp_path / "traces.sqlite").exists()


@pytest.mark.asyncio
async def test_proxy_injects_cc_segment_separators_for_cc_requests(tmp_path, monkeypatch):
    """when a CC-shaped request hits /v1/messages, the proxy
    rewrites the body to inject ` # # ` separators around recognized
    structural blocks BEFORE forwarding to the backend. This is the wiring
    that lets cacheblend's segment detector find chunks in natural CC
    traffic."""
    backend_port = _free_port()
    proxy_port = _free_port()

    monkeypatch.setenv("SKILLCACHER_BACKEND_URL", f"http://127.0.0.1:{backend_port}")
    monkeypatch.setenv("SKILLCACHER_PROXY_PORT", str(proxy_port))
    monkeypatch.setenv("SKILLCACHER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("SKILLCACHER_TOKENIZER_NAME", "NousResearch/Meta-Llama-3-8B")
    monkeypatch.setenv("SKILLCACHER_SPAN_REGISTRY_PATH", str(tmp_path / "span_registry.sqlite"))
    monkeypatch.setenv("SKILLCACHER_SKILL_DIRS", "")
    monkeypatch.setenv("SKILLCACHER_ENABLE_PRE_SEED", "false")
    monkeypatch.setenv("SKILLCACHER_ENABLE_STDOUT_TAIL", "false")
    monkeypatch.setenv("SKILLCACHER_CC_SEGMENT_PARSER", "true")

    captured_bodies: list[dict] = []

    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat(req: dict):
        captured_bodies.append(req)
        return {
            "id": "x", "object": "chat.completion", "model": req["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        }

    from skillcacher.proxy.server import build_app
    proxy = build_app()

    cc_request = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "system": (
            "x-anthropic-billing-header: cc_version=2.1.136.829; cch=abc;"
            "You are a Claude agent.CWD: /repo\nDate: 2026-05-08"
        ),
        "messages": [
            {"role": "user", "content": (
                "<system-reminder>\nctx note\n</system-reminder>\n\n"
                "Now answer my question about asyncio."
            )},
        ],
    }

    with _serve(backend, backend_port), _serve(proxy, proxy_port):
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"http://127.0.0.1:{proxy_port}/v1/messages", json=cc_request,
            )
    assert r.status_code == 200, r.text
    assert captured_bodies, "backend never received the request"

    # The backend got a translated OpenAI-shaped request. Concatenate every
    # message content to inspect what cacheblend would have seen.
    bb = captured_bodies[0]
    full = "".join(
        m.get("content", "") for m in bb["messages"] if isinstance(m.get("content"), str)
    )
    assert " # # " in full, (
        "expected ` # # ` separator in backend-received messages; got: " + full[:300]
    )


@pytest.mark.asyncio
async def test_proxy_skips_cc_segment_rewrite_when_disabled(tmp_path, monkeypatch):
    """The parser is flag-gated. When SKILLCACHER_CC_SEGMENT_PARSER=false,
    the body forwards verbatim — useful for A/B comparing the capture
    pre-parser baseline against the post-parser numbers."""
    backend_port = _free_port()
    proxy_port = _free_port()

    monkeypatch.setenv("SKILLCACHER_BACKEND_URL", f"http://127.0.0.1:{backend_port}")
    monkeypatch.setenv("SKILLCACHER_PROXY_PORT", str(proxy_port))
    monkeypatch.setenv("SKILLCACHER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("SKILLCACHER_TOKENIZER_NAME", "NousResearch/Meta-Llama-3-8B")
    monkeypatch.setenv("SKILLCACHER_SPAN_REGISTRY_PATH", str(tmp_path / "span_registry.sqlite"))
    monkeypatch.setenv("SKILLCACHER_SKILL_DIRS", "")
    monkeypatch.setenv("SKILLCACHER_ENABLE_PRE_SEED", "false")
    monkeypatch.setenv("SKILLCACHER_ENABLE_STDOUT_TAIL", "false")
    monkeypatch.setenv("SKILLCACHER_CC_SEGMENT_PARSER", "false")

    captured_bodies: list[dict] = []
    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat(req: dict):
        captured_bodies.append(req)
        return {
            "id": "x", "object": "chat.completion", "model": req["model"],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 1, "total_tokens": 8},
        }

    from skillcacher.proxy.server import build_app
    proxy = build_app()

    cc_request = {
        "model": "claude-sonnet-4",
        "max_tokens": 16,
        "system": "x-anthropic-billing-header: cc_version=2.1.136.829; cch=abc;CWD: /r\nDate: 2026-05-08",
        "messages": [{"role": "user", "content": "<system-reminder>\nctx\n</system-reminder>"}],
    }

    with _serve(backend, backend_port), _serve(proxy, proxy_port):
        async with httpx.AsyncClient(timeout=10) as client:
            r = await client.post(
                f"http://127.0.0.1:{proxy_port}/v1/messages", json=cc_request,
            )
    assert r.status_code == 200, r.text
    assert captured_bodies
    bb = captured_bodies[0]
    full = "".join(
        m.get("content", "") for m in bb["messages"] if isinstance(m.get("content"), str)
    )
    # Disabled: separator is NOT injected.
    assert " # # " not in full, (
        "separator should NOT be present when SKILLCACHER_CC_SEGMENT_PARSER=false: "
        + full[:300]
    )


@pytest.mark.asyncio
async def test_proxy_streaming_tool_calls(tmp_path, monkeypatch):
    backend_port = _free_port()
    proxy_port = _free_port()

    monkeypatch.setenv("SKILLCACHER_BACKEND_URL", f"http://127.0.0.1:{backend_port}")
    monkeypatch.setenv("SKILLCACHER_PROXY_PORT", str(proxy_port))
    monkeypatch.setenv("SKILLCACHER_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("SKILLCACHER_TOKENIZER_NAME", "NousResearch/Meta-Llama-3-8B")
    monkeypatch.setenv("SKILLCACHER_SPAN_REGISTRY_PATH", str(tmp_path / "span_registry.sqlite"))
    monkeypatch.setenv("SKILLCACHER_SKILL_DIRS", "")
    monkeypatch.setenv("SKILLCACHER_ENABLE_PRE_SEED", "false")
    monkeypatch.setenv("SKILLCACHER_ENABLE_STDOUT_TAIL", "false")

    backend = FastAPI()

    @backend.post("/v1/chat/completions")
    async def chat(req: dict):
        from fastapi.responses import StreamingResponse as SR
        async def gen():
            yield 'data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
            yield 'data: {"id":"x","object":"chat.completion.chunk","model":"m","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":3,"total_tokens":15}}\n\n'
            yield 'data: [DONE]\n\n'
        return SR(gen(), media_type="text/event-stream")

    from skillcacher.proxy.server import build_app
    proxy = build_app()

    with _serve(backend, backend_port), _serve(proxy, proxy_port):
        req = json.loads(FIXTURE.read_text())
        req["stream"] = True
        async with httpx.AsyncClient(timeout=10) as client:
            async with client.stream("POST", f"http://127.0.0.1:{proxy_port}/v1/messages", json=req) as r:
                events: list[dict] = []
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        payload = line[len("data:"):].strip()
                        if payload and payload != "[DONE]":
                            events.append(json.loads(payload))

    # Find the message_delta event
    deltas = [e for e in events if e.get("type") == "message_delta"]
    assert len(deltas) == 1
    assert deltas[0]["delta"]["stop_reason"] == "tool_use"
    assert deltas[0]["usage"]["input_tokens"] == 12
    assert deltas[0]["usage"]["output_tokens"] == 3
    # Confirm no [DONE] trailer in the events list
    # (Already filtered above; this is implicit.)

    # Confirm the trace was written even though the stream ended successfully
    # (post-yield trace_store.write must have run).
    assert (tmp_path / "traces.sqlite").exists()
