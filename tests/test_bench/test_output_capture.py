"""Tests for bench/output_capture.py — replay-with-output-capture.

Uses an injected stub client so the tests run offline; no httpx server,
no live proxy."""
from __future__ import annotations

import asyncio

import pytest

from skillcacher.bench.output_capture import (
    Generation,
    _extract_text,
    replay_with_output_capture,
)
from skillcacher.settings import Settings


class _StubResponse:
    def __init__(self, json_data: dict, status_code: int = 200):
        self._json = json_data
        self.status_code = status_code

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _StubClient:
    def __init__(self, response: _StubResponse):
        self._response = response
        self.last_call: dict | None = None

    async def post(self, url: str, json: dict | None = None):  # noqa: A002
        self.last_call = {"url": url, "json": json}
        return self._response


def _anth_response(text: str = "hello world",
                   tool_use: dict | None = None,
                   stop_reason: str = "end_turn") -> dict:
    content: list[dict] = [{"type": "text", "text": text}]
    if tool_use is not None:
        content.append(tool_use)
    return {
        "id": "msg_test_001",
        "type": "message",
        "role": "assistant",
        "model": "test-model",
        "content": content,
        "stop_reason": stop_reason,
        "usage": {"input_tokens": 12, "output_tokens": 5},
    }


def test_extract_text_joins_text_blocks_only():
    blocks = [
        {"type": "text", "text": "hello "},
        {"type": "tool_use", "name": "x", "input": {}},
        {"type": "text", "text": "world"},
    ]
    assert _extract_text(blocks) == "hello world"


def test_extract_text_handles_empty_or_missing():
    assert _extract_text([]) == ""
    assert _extract_text([{"type": "text"}]) == ""


def test_replay_returns_generation_for_text_response():
    client = _StubClient(_StubResponse(_anth_response("hi there")))
    settings = Settings()
    gen = asyncio.run(replay_with_output_capture(
        {"messages": [{"role": "user", "content": "hi"}]},
        settings, client=client, proxy_url="http://stub/v1/messages",
    ))
    assert isinstance(gen, Generation)
    assert gen.text == "hi there"
    assert gen.stop_reason == "end_turn"
    assert gen.input_tokens == 12
    assert gen.output_tokens == 5
    assert gen.response_id == "msg_test_001"
    assert gen.ttft_ms >= 0.0
    assert len(gen.content_blocks) == 1


def test_replay_forces_stream_false_even_if_request_streamed():
    client = _StubClient(_StubResponse(_anth_response()))
    settings = Settings()
    asyncio.run(replay_with_output_capture(
        {"messages": [{"role": "user", "content": "x"}], "stream": True},
        settings, client=client, proxy_url="http://stub/v1/messages",
    ))
    assert client.last_call is not None
    assert client.last_call["json"]["stream"] is False


def test_replay_temperature_override_wins_over_request_body():
    client = _StubClient(_StubResponse(_anth_response()))
    settings = Settings()
    asyncio.run(replay_with_output_capture(
        {"messages": [{"role": "user", "content": "x"}], "temperature": 0.9},
        settings, client=client, proxy_url="http://stub/v1/messages",
        temperature=0.0,
    ))
    assert client.last_call["json"]["temperature"] == 0.0


def test_replay_temperature_omitted_preserves_request_body_value():
    client = _StubClient(_StubResponse(_anth_response()))
    settings = Settings()
    asyncio.run(replay_with_output_capture(
        {"messages": [{"role": "user", "content": "x"}], "temperature": 0.7},
        settings, client=client, proxy_url="http://stub/v1/messages",
    ))
    assert client.last_call["json"]["temperature"] == 0.7


def test_replay_does_not_mutate_input_request_body():
    client = _StubClient(_StubResponse(_anth_response()))
    settings = Settings()
    body = {"messages": [{"role": "user", "content": "x"}], "stream": True}
    body_before = {**body}
    asyncio.run(replay_with_output_capture(
        body, settings, client=client, proxy_url="http://stub/v1/messages",
    ))
    assert body == body_before  # caller's dict untouched


def test_replay_handles_tool_use_block():
    tool_use = {"type": "tool_use", "id": "t1", "name": "search",
                "input": {"q": "weather", "n": 3}}
    client = _StubClient(_StubResponse(_anth_response("calling tool", tool_use=tool_use)))
    settings = Settings()
    gen = asyncio.run(replay_with_output_capture(
        {"messages": []}, settings, client=client, proxy_url="http://stub/v1/messages",
    ))
    assert len(gen.content_blocks) == 2
    assert gen.content_blocks[1]["type"] == "tool_use"
    assert gen.text == "calling tool"  # text-only extraction


def test_replay_handles_missing_usage_and_stop_reason():
    client = _StubClient(_StubResponse({"id": "msg_x", "content": []}))
    settings = Settings()
    gen = asyncio.run(replay_with_output_capture(
        {"messages": []}, settings, client=client, proxy_url="http://stub/v1/messages",
    ))
    assert gen.input_tokens == 0
    assert gen.output_tokens == 0
    assert gen.stop_reason == ""
    assert gen.text == ""


def test_replay_raises_on_http_error():
    client = _StubClient(_StubResponse({}, status_code=502))
    settings = Settings()
    with pytest.raises(RuntimeError):
        asyncio.run(replay_with_output_capture(
            {"messages": []}, settings, client=client, proxy_url="http://stub/v1/messages",
        ))
