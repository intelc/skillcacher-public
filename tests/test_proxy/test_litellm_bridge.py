"""Tests for the LiteLLM bridge — Anthropic Messages request → litellm
acompletion call → Anthropic-shaped response.

We mock litellm.acompletion to keep tests offline."""
import pytest
from unittest.mock import patch, AsyncMock

from skillcacher.proxy.litellm_bridge import (
    anthropic_to_litellm_kwargs,
    litellm_to_anthropic_response,
    call_unary,
    stream_to_anthropic_events,
    call_streaming,
)


def test_anthropic_to_litellm_kwargs_basic():
    req = {
        "model": "claude-3-5-sonnet-20241022",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32,
    }
    kw = anthropic_to_litellm_kwargs(req, served_model="Qwen/Qwen3-8B", api_base="http://x", api_key="k")
    assert kw["model"] == "hosted_vllm/Qwen/Qwen3-8B"
    assert kw["api_base"] == "http://x"
    assert kw["api_key"] == "k"
    assert kw["max_tokens"] == 32
    assert kw["messages"] == [{"role": "user", "content": "hi"}]


def test_anthropic_to_litellm_kwargs_preserves_system_block():
    req = {
        "model": "claude-3-5-sonnet-20241022",
        "system": [{"type": "text", "text": "you are helpful"}],
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 32,
    }
    kw = anthropic_to_litellm_kwargs(req, served_model="m", api_base="x", api_key="k")
    assert kw["messages"][0] == {"role": "system", "content": "you are helpful"}
    assert kw["messages"][1] == {"role": "user", "content": "hi"}


def test_anthropic_to_litellm_kwargs_drops_top_k():
    req = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1, "top_k": 5}
    kw = anthropic_to_litellm_kwargs(req, served_model="m", api_base="x", api_key="k")
    assert "top_k" not in kw


def test_anthropic_to_litellm_kwargs_passes_tools():
    tools = [
        {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}},
        }
    ]
    req = {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 1, "tools": tools}
    kw = anthropic_to_litellm_kwargs(req, served_model="m", api_base="x", api_key="k")
    assert kw["tools"][0]["type"] == "function"
    assert kw["tools"][0]["function"]["name"] == "get_weather"
    assert kw["tools"][0]["function"]["parameters"] == tools[0]["input_schema"]


def test_litellm_to_anthropic_response_text():
    # Simulate a litellm.ModelResponse-shaped dict
    lr = {
        "id": "chatcmpl-abc",
        "model": "Qwen/Qwen3-8B",
        "choices": [
            {
                "message": {"role": "assistant", "content": "hello"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 2},
    }
    out = litellm_to_anthropic_response(lr, original_model="claude-3-5-sonnet-20241022")
    assert out["type"] == "message"
    assert out["role"] == "assistant"
    assert out["model"] == "claude-3-5-sonnet-20241022"
    assert out["stop_reason"] == "end_turn"
    assert out["content"][0]["type"] == "text"
    assert out["content"][0]["text"] == "hello"
    assert out["usage"]["input_tokens"] == 10
    assert out["usage"]["output_tokens"] == 2


def test_litellm_to_anthropic_response_tool_use():
    lr = {
        "id": "chatcmpl-x",
        "model": "m",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 8},
    }
    out = litellm_to_anthropic_response(lr, original_model="claude-3-5-sonnet-20241022")
    assert out["stop_reason"] == "tool_use"
    assert out["content"][0]["type"] == "tool_use"
    assert out["content"][0]["name"] == "get_weather"
    assert out["content"][0]["input"] == {"city": "Tokyo"}
    assert out["content"][0]["id"] == "call_1"


def test_translate_user_message_with_tool_results_flattens_to_tool_messages():
    req = {
        "model": "claude-x",
        "messages": [
            {"role": "assistant", "content": [
                {"type": "tool_use", "id": "call_1", "name": "get_weather", "input": {"city": "Tokyo"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "Sunny, 22C"},
                {"type": "text", "text": "Thanks. What about Osaka?"},
            ]},
        ],
        "max_tokens": 32,
    }
    kw = anthropic_to_litellm_kwargs(req, served_model="m", api_base="x", api_key="k")
    msgs = kw["messages"]
    # Expect: assistant(tool_calls), tool(call_1), user("Thanks...")
    assert msgs[0]["role"] == "assistant"
    assert msgs[0]["tool_calls"][0]["id"] == "call_1"
    assert msgs[1]["role"] == "tool"
    assert msgs[1]["tool_call_id"] == "call_1"
    assert msgs[1]["content"] == "Sunny, 22C"
    assert msgs[2]["role"] == "user"
    assert "Osaka" in msgs[2]["content"]
    # No invalid roles
    for m in msgs:
        assert m["role"] in {"system", "user", "assistant", "tool"}


def test_translate_user_message_tool_results_only_no_text():
    """When the user message has only tool_results and no text, no trailing user message."""
    req = {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "ok"},
            ]},
        ],
        "max_tokens": 1,
    }
    kw = anthropic_to_litellm_kwargs(req, served_model="m", api_base="x", api_key="k")
    msgs = kw["messages"]
    assert len(msgs) == 1
    assert msgs[0]["role"] == "tool"
    assert msgs[0]["tool_call_id"] == "call_1"
    assert msgs[0]["content"] == "ok"


def test_translate_message_tool_result_with_list_content():
    """tool_result.content can be a list of {"type": "text", "text": "..."} blocks."""
    req = {
        "model": "claude-x",
        "messages": [
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "call_1",
                 "content": [
                     {"type": "text", "text": "line 1"},
                     {"type": "text", "text": "line 2"},
                 ]},
            ]},
        ],
        "max_tokens": 1,
    }
    kw = anthropic_to_litellm_kwargs(req, served_model="m", api_base="x", api_key="k")
    msgs = kw["messages"]
    assert msgs[0]["role"] == "tool"
    assert "line 1" in msgs[0]["content"]
    assert "line 2" in msgs[0]["content"]


@pytest.mark.asyncio
async def test_call_unary_invokes_litellm():
    fake_resp = type(
        "R", (), {"model_dump": lambda self: {
            "id": "x", "model": "m",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }}
    )()
    with patch("skillcacher.proxy.litellm_bridge.litellm.acompletion", new=AsyncMock(return_value=fake_resp)) as m:
        out, raw = await call_unary(
            req={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            served_model="m", api_base="http://x", api_key="k",
        )
    assert m.called
    assert out["content"][0]["text"] == "ok"
    assert raw["id"] == "x"


async def _async_iter(items):
    for it in items:
        yield it


def _chunk(delta_content=None, delta_tool_calls=None, finish_reason=None, usage=None):
    """Build a litellm-shaped stream chunk."""
    return type("C", (), {"model_dump": lambda self: {
        "id": "x",
        "choices": [{
            "delta": {"content": delta_content, "tool_calls": delta_tool_calls},
            "finish_reason": finish_reason,
        }],
        "usage": usage,
    }})()


@pytest.mark.asyncio
async def test_stream_text_emits_anthropic_events():
    chunks = [
        _chunk(delta_content="hel"),
        _chunk(delta_content="lo"),
        _chunk(finish_reason="stop", usage={"prompt_tokens": 5, "completion_tokens": 2}),
    ]
    events = []
    async for ev in stream_to_anthropic_events(_async_iter(chunks), original_model="claude-x"):
        events.append(ev)

    types = [e["type"] for e in events]
    assert types[0] == "message_start"
    assert "content_block_start" in types
    assert any(e["type"] == "content_block_delta" and e["delta"]["text"] == "hel" for e in events)
    assert any(e["type"] == "content_block_delta" and e["delta"]["text"] == "lo" for e in events)
    assert types[-2:] == ["message_delta", "message_stop"]
    msg_delta = next(e for e in events if e["type"] == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "end_turn"
    assert msg_delta["usage"]["input_tokens"] == 5
    assert msg_delta["usage"]["output_tokens"] == 2


@pytest.mark.asyncio
async def test_stream_tool_calls_emits_tool_use_blocks():
    chunks = [
        _chunk(delta_tool_calls=[{
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "get_weather", "arguments": ""},
        }]),
        _chunk(delta_tool_calls=[{
            "index": 0, "function": {"arguments": '{"city":'},
        }]),
        _chunk(delta_tool_calls=[{
            "index": 0, "function": {"arguments": '"Tokyo"}'},
        }]),
        _chunk(finish_reason="tool_calls", usage={"prompt_tokens": 5, "completion_tokens": 8}),
    ]
    events = []
    async for ev in stream_to_anthropic_events(_async_iter(chunks), original_model="claude-x"):
        events.append(ev)

    starts = [e for e in events if e["type"] == "content_block_start"]
    assert any(s["content_block"]["type"] == "tool_use" for s in starts)
    tool_start = next(s for s in starts if s["content_block"]["type"] == "tool_use")
    assert tool_start["content_block"]["name"] == "get_weather"
    assert tool_start["content_block"]["id"] == "call_1"
    deltas = [e for e in events if e["type"] == "content_block_delta"]
    arg_deltas = [d for d in deltas if d["delta"].get("type") == "input_json_delta"]
    full_args = "".join(d["delta"]["partial_json"] for d in arg_deltas)
    assert full_args == '{"city":"Tokyo"}'
    msg_delta = next(e for e in events if e["type"] == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "tool_use"


@pytest.mark.asyncio
async def test_stream_empty_chunks_still_emits_terminal_envelope():
    events = []
    async for ev in stream_to_anthropic_events(_async_iter([]), original_model="claude-x"):
        events.append(ev)
    types = [e["type"] for e in events]
    # message_start, then no blocks opened, then message_delta, then message_stop
    assert types == ["message_start", "message_delta", "message_stop"]
    msg_delta = next(e for e in events if e["type"] == "message_delta")
    assert msg_delta["delta"]["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_stream_interleaved_text_and_tool_calls_close_in_index_order():
    chunks = [
        _chunk(delta_content="hi"),                                     # text block opened at index 0
        _chunk(delta_tool_calls=[{                                       # tool_use opened at index 1
            "index": 0, "id": "call_1", "type": "function",
            "function": {"name": "f", "arguments": "{}"},
        }]),
        _chunk(finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1}),
    ]
    events = []
    async for ev in stream_to_anthropic_events(_async_iter(chunks), original_model="claude-x"):
        events.append(ev)
    stops = [e for e in events if e["type"] == "content_block_stop"]
    indices = [s["index"] for s in stops]
    assert indices == sorted(indices), f"stops out of order: {indices}"


@pytest.mark.asyncio
async def test_call_streaming_invokes_litellm_with_stream_options():
    async def fake_stream():
        yield _chunk(delta_content="ok")
        yield _chunk(finish_reason="stop", usage={"prompt_tokens": 1, "completion_tokens": 1})

    captured_kwargs = {}

    async def fake_acompletion(**kwargs):
        captured_kwargs.update(kwargs)
        return fake_stream()

    with patch("skillcacher.proxy.litellm_bridge.litellm.acompletion", new=fake_acompletion):
        events = []
        async for ev in call_streaming(
            req={"model": "claude-x", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            served_model="m", api_base="http://x", api_key="k",
        ):
            events.append(ev)
    assert captured_kwargs["stream"] is True
    assert captured_kwargs["stream_options"] == {"include_usage": True}
    assert any(e["type"] == "content_block_delta" and e["delta"].get("text") == "ok" for e in events)


def test_normalize_api_base_appends_v1():
    from skillcacher.proxy.litellm_bridge import _normalize_api_base
    assert _normalize_api_base("http://host:8000") == "http://host:8000/v1"


def test_normalize_api_base_idempotent_when_already_v1():
    from skillcacher.proxy.litellm_bridge import _normalize_api_base
    assert _normalize_api_base("http://host:8000/v1") == "http://host:8000/v1"


def test_normalize_api_base_strips_trailing_slash():
    from skillcacher.proxy.litellm_bridge import _normalize_api_base
    assert _normalize_api_base("http://host:8000/v1/") == "http://host:8000/v1"


def test_normalize_api_base_strips_trailing_slash_without_v1():
    from skillcacher.proxy.litellm_bridge import _normalize_api_base
    assert _normalize_api_base("http://host:8000/") == "http://host:8000/v1"
