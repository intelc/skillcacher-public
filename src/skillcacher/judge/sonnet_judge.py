"""Judge: Sonnet-4.6 via official Anthropic SDK.

Uses prompt caching on the static judge system prompt (deterministic across
all calls in a batch). The user message varies per pair, so we don't cache
it. With ~47 pairs at ~600-byte system prompt and ~500-2000 byte user
messages, this is small enough that caching mainly helps with consistency
rather than cost — but the SDK breakpoint is free to add."""
from __future__ import annotations

import logging
import os
import time

import anthropic

from skillcacher.judge.prompt import JUDGE_SYSTEM, JudgeCall, JudgeResult, parse_response

log = logging.getLogger("skillcacher.judge")

DEFAULT_JUDGE_MODEL = "claude-sonnet-4-6"
DEFAULT_MAX_TOKENS = 200


def make_client(api_key: str | None = None) -> anthropic.Anthropic:
    """Construct an Anthropic client. Picks up ANTHROPIC_API_KEY from env if
    not passed explicitly."""
    return anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))


def call_judge(
    call: JudgeCall,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> JudgeResult:
    """Submit one pair to the judge and parse its response.

    Caches the (static) system prompt via top-level `cache_control` so a
    batch of consecutive judge calls re-uses the prefix. With ~600-byte
    system prompts this only crosses the cache threshold on Sonnet-4.6 if
    the request includes substantial user content too — that's typical for
    these pairs (~1-4KB of agent outputs), so caching often kicks in."""
    if client is None:
        client = make_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{"type": "text", "text": JUDGE_SYSTEM,
                 "cache_control": {"type": "ephemeral"}}],
        messages=[{"role": "user", "content": call.user_message}],
    )
    raw = ""
    for block in response.content:
        if block.type == "text":
            raw += block.text
    return parse_response(call, raw)


def call_judge_with_retry(
    call: JudgeCall,
    *,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_JUDGE_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_retries: int = 3,
    base_delay: float = 1.0,
) -> JudgeResult:
    """Same as call_judge but retries on rate limits + transient errors.

    The Anthropic SDK retries automatically by default (max_retries=2 on
    the client); this adds a wrapper for retries that should also re-call
    even on unparseable responses (where the model returned text but no
    valid label). Unparseable retries are bounded so a model that ALWAYS
    misformats can't loop forever."""
    last: JudgeResult | None = None
    for attempt in range(max_retries):
        try:
            result = call_judge(call, client=client, model=model, max_tokens=max_tokens)
        except anthropic.RateLimitError:
            delay = base_delay * (2 ** attempt)
            log.warning("judge rate-limited; sleeping %.1fs (attempt %d/%d)",
                        delay, attempt + 1, max_retries)
            time.sleep(delay)
            continue
        except anthropic.APIStatusError as e:
            if e.status_code >= 500 and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                log.warning("judge %dxx; retrying in %.1fs", e.status_code, delay)
                time.sleep(delay)
                continue
            raise
        if result.label != "UNPARSEABLE":
            return result
        last = result
        log.warning("judge returned UNPARSEABLE label (attempt %d/%d): %r",
                    attempt + 1, max_retries, result.raw_response[:120])
    return last if last is not None else JudgeResult(
        pair=call.pair, position_a=call.position_a,
        raw_response="", label="UNPARSEABLE", prefers="unparseable", rationale="",
    )
