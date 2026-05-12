"""Bridge between Anthropic Messages API requests and LiteLLM's
acompletion call against an OpenAI-compatible vLLM backend.

We delegate Anthropic↔OpenAI translation to LiteLLM via the `hosted_vllm/`
provider prefix; this preserves cache_control and tool_use schema
correctness without us owning the translation surface.

The proxy's FastAPI handler wraps these helpers so we keep request /
response intercepts for trace-store writes and Controller.Lookup calls.
"""
from __future__ import annotations

import json
import uuid
from typing import Any, AsyncIterator

import litellm

ANTHROPIC_STOP_MAP = {
    "stop": "end_turn",
    "tool_calls": "tool_use",
    "length": "max_tokens",
    "content_filter": "end_turn",
}


def anthropic_to_litellm_kwargs(
    req: dict[str, Any], *, served_model: str, api_base: str, api_key: str,
    max_completion_tokens: int | None = None,
) -> dict[str, Any]:
    """Build kwargs for litellm.acompletion from an Anthropic Messages request.

    LiteLLM owns the schema mapping; we only flatten Anthropic's split
    `system` / `messages` representation into a single OpenAI-shaped
    `messages` list and translate Anthropic-shaped tools into OpenAI
    function tools."""
    messages: list[dict[str, Any]] = []
    sys_b = req.get("system")
    if isinstance(sys_b, str):
        messages.append({"role": "system", "content": sys_b})
    elif isinstance(sys_b, list):
        text = "\n".join(b.get("text", "") for b in sys_b if b.get("type") == "text")
        if text:
            messages.append({"role": "system", "content": text})

    for m in req.get("messages") or []:
        messages.extend(_translate_message(m))

    tools = None
    if req.get("tools"):
        tools = [_anthropic_tool_to_openai(t) for t in req["tools"]]

    requested_max = req.get("max_tokens", 1024)
    effective_max = (
        min(requested_max, max_completion_tokens)
        if max_completion_tokens
        else requested_max
    )
    kwargs: dict[str, Any] = {
        "model": f"hosted_vllm/{served_model}",
        "api_base": api_base,
        "api_key": api_key or "EMPTY",
        "messages": messages,
        "max_tokens": effective_max,
    }
    for k in ("temperature", "top_p", "stop"):
        if k in req:
            kwargs[k] = req[k]
    if tools is not None:
        kwargs["tools"] = tools
        if "tool_choice" in req:
            kwargs["tool_choice"] = req["tool_choice"]
    # NB: top_k is intentionally dropped — OpenAI-compatible backends reject it.
    return kwargs


def _translate_message(m: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate one Anthropic message into one OR MORE OpenAI messages.

    Always returns a list (even for the single-message case). A `user`
    message containing tool_result blocks expands into:
      [tool message per tool_result] + [optional user text continuation]
    OpenAI requires tool messages to follow the prior assistant's
    tool_calls before any further user text."""
    role = m.get("role", "user")
    c = m.get("content")
    if isinstance(c, str):
        return [{"role": role, "content": c}]
    if not isinstance(c, list):
        return [{"role": role, "content": ""}]

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    tool_results: list[dict[str, Any]] = []
    for b in c:
        t = b.get("type")
        if t == "text":
            text_parts.append(b.get("text", ""))
        elif t == "tool_use":
            tool_calls.append({
                "id": b.get("id", ""),
                "type": "function",
                "function": {
                    "name": b.get("name", ""),
                    "arguments": json.dumps(b.get("input", {})),
                },
            })
        elif t == "tool_result":
            content = b.get("content", "")
            if isinstance(content, list):
                content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
            tool_results.append({
                "role": "tool",
                "tool_call_id": b.get("tool_use_id", ""),
                "content": str(content),
            })

    if tool_results and role == "user":
        # tool messages first (must follow prior assistant's tool_calls),
        # then user text continuation if any.
        out_msgs: list[dict[str, Any]] = list(tool_results)
        text = "\n".join(text_parts).strip()
        if text:
            out_msgs.append({"role": "user", "content": text})
        return out_msgs

    msg: dict[str, Any] = {"role": role}
    if text_parts:
        msg["content"] = "\n".join(text_parts)
    if tool_calls:
        # OpenAI: assistant message with tool_calls may have null content
        msg["content"] = msg.get("content")  # explicit None if no text
        msg["tool_calls"] = tool_calls
    return [msg]


def _anthropic_tool_to_openai(tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("input_schema", {"type": "object"}),
        },
    }


def litellm_to_anthropic_response(lr: dict[str, Any], *, original_model: str) -> dict[str, Any]:
    """Translate a litellm completion dict into an Anthropic Messages response.

    Handles both text-only completions and tool_calls completions."""
    choice = lr["choices"][0]
    msg = choice.get("message", {})
    content_blocks: list[dict[str, Any]] = []
    if (text := msg.get("content")):
        content_blocks.append({"type": "text", "text": text})
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id", ""),
            "name": fn.get("name", ""),
            "input": args,
        })
    finish_reason = choice.get("finish_reason", "stop")
    usage = lr.get("usage") or {}
    return {
        "id": lr.get("id", ""),
        "type": "message",
        "role": "assistant",
        "model": original_model,
        "content": content_blocks,
        "stop_reason": ANTHROPIC_STOP_MAP.get(finish_reason, "end_turn"),
        "stop_sequence": None,
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }


def _normalize_api_base(api_base: str) -> str:
    """vLLM serves OpenAI endpoints under `/v1`. LiteLLM's hosted_vllm
    provider appends `/chat/completions` directly to `api_base`, so we
    must ensure the base ends with `/v1` here. Settings.backend_url is
    intentionally stored as the host root (no `/v1`) for compatibility
    with the original direct-httpx code path; we pin the convention here."""
    base = api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"
    return base


async def call_unary(
    req: dict[str, Any], *, served_model: str, api_base: str, api_key: str,
    max_completion_tokens: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Invoke litellm.acompletion (non-streaming) and return both the
    Anthropic-shaped response and the raw litellm dict (for trace logging)."""
    kwargs = anthropic_to_litellm_kwargs(
        req, served_model=served_model, api_base=_normalize_api_base(api_base), api_key=api_key,
        max_completion_tokens=max_completion_tokens,
    )
    response = await litellm.acompletion(**kwargs)
    raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    out = litellm_to_anthropic_response(raw, original_model=req.get("model", ""))
    return out, raw


async def stream_to_anthropic_events(
    chunks: AsyncIterator[Any], *, original_model: str,
) -> AsyncIterator[dict[str, Any]]:
    """Translate a stream of LiteLLM completion chunks into Anthropic SSE events.

    Manages content-block lifecycle: opens a `text` block on the first
    content delta, opens a `tool_use` block on the first tool_call delta
    (with index = block index), accumulates argument deltas as
    `input_json_delta`, closes blocks at finish, emits `message_delta` +
    `message_stop` with usage."""
    msg_id = "msg_" + uuid.uuid4().hex[:16]
    yield {
        "type": "message_start",
        "message": {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": [],
            "model": original_model,
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        },
    }

    text_block_index: int | None = None
    tool_blocks: dict[int, int] = {}  # tool_call_index → block_index
    next_block_index = 0
    usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}
    finish_reason: str | None = None

    async for chunk in chunks:
        c = chunk.model_dump() if hasattr(chunk, "model_dump") else dict(chunk)
        if (u := c.get("usage")):
            # LiteLLM emits usage on the terminal chunk with OpenAI keys;
            # only adopt if the keys we need are actually present, to avoid
            # silently overwriting a final usage with a partial intermediate.
            if "prompt_tokens" in u or "completion_tokens" in u:
                usage = u
        choices = c.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        if (text := delta.get("content")):
            if text_block_index is None:
                text_block_index = next_block_index
                next_block_index += 1
                yield {
                    "type": "content_block_start",
                    "index": text_block_index,
                    "content_block": {"type": "text", "text": ""},
                }
            yield {
                "type": "content_block_delta",
                "index": text_block_index,
                "delta": {"type": "text_delta", "text": text},
            }
        for tc in delta.get("tool_calls") or []:
            tc_idx = tc.get("index", 0)
            if tc_idx not in tool_blocks:
                block_idx = next_block_index
                next_block_index += 1
                tool_blocks[tc_idx] = block_idx
                fn = tc.get("function") or {}
                yield {
                    "type": "content_block_start",
                    "index": block_idx,
                    "content_block": {
                        "type": "tool_use",
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "input": {},
                    },
                }
            block_idx = tool_blocks[tc_idx]
            fn = tc.get("function") or {}
            args_partial = fn.get("arguments")
            if args_partial:
                yield {
                    "type": "content_block_delta",
                    "index": block_idx,
                    "delta": {"type": "input_json_delta", "partial_json": args_partial},
                }
        if (fr := choices[0].get("finish_reason")):
            finish_reason = fr

    # Close all open blocks in ascending index order (Anthropic SDK
    # convention: stop events arrive in the same order as block_start).
    open_indices: list[int] = list(tool_blocks.values())
    if text_block_index is not None:
        open_indices.append(text_block_index)
    for idx in sorted(open_indices):
        yield {"type": "content_block_stop", "index": idx}

    yield {
        "type": "message_delta",
        "delta": {
            "stop_reason": ANTHROPIC_STOP_MAP.get(finish_reason or "stop", "end_turn"),
            "stop_sequence": None,
        },
        "usage": {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
        },
    }
    yield {"type": "message_stop"}


async def call_streaming(
    req: dict[str, Any], *, served_model: str, api_base: str, api_key: str,
    max_completion_tokens: int | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Invoke litellm.acompletion (streaming) and yield Anthropic SSE events."""
    kwargs = anthropic_to_litellm_kwargs(
        req, served_model=served_model, api_base=_normalize_api_base(api_base), api_key=api_key,
        max_completion_tokens=max_completion_tokens,
    )
    kwargs["stream"] = True
    kwargs["stream_options"] = {"include_usage": True}
    stream = await litellm.acompletion(**kwargs)
    async for ev in stream_to_anthropic_events(stream, original_model=req.get("model", "")):
        yield ev
