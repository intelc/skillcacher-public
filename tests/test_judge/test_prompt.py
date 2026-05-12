"""Tests for judge/prompt.py — render_call + parse_response."""
from __future__ import annotations

import random

from skillcacher.judge.prompt import JudgePair, parse_response, render_call


def _pair(cb: str = "cacheblend output", nc: str = "no_cache output",
          prompt: str = "do the thing") -> JudgePair:
    return JudgePair(
        capture_id="cap_x", turn_index=0,
        cacheblend_text=cb, no_cache_text=nc, task_prompt=prompt,
    )


def test_render_call_assigns_position_via_rng():
    pair = _pair()
    # rng with random()<0.5 → cb is A
    rng_low = random.Random()
    rng_low.random = lambda: 0.1  # type: ignore
    call = render_call(pair, rng=rng_low)
    assert call.position_a == "cacheblend"
    assert "cacheblend output" in call.user_message
    # Position-of-A in the rendered text is before position-of-B
    assert call.user_message.index("cacheblend output") < call.user_message.index("no_cache output")


def test_render_call_random_high_swaps_position():
    pair = _pair()
    rng_high = random.Random()
    rng_high.random = lambda: 0.9  # type: ignore
    call = render_call(pair, rng=rng_high)
    assert call.position_a == "no_cache"
    assert call.user_message.index("no_cache output") < call.user_message.index("cacheblend output")


def test_render_call_includes_task_prompt():
    pair = _pair(prompt="please refactor the function")
    call = render_call(pair, rng=random.Random(0))
    assert "please refactor the function" in call.user_message


def test_render_call_handles_empty_task_prompt():
    pair = _pair(prompt="")
    call = render_call(pair, rng=random.Random(0))
    assert "(none provided)" in call.user_message


def test_render_call_handles_empty_outputs():
    pair = _pair(cb="", nc="")
    call = render_call(pair, rng=random.Random(0))
    assert call.user_message.count("(empty)") == 2


def test_parse_response_prefer_a_with_cb_at_a_resolves_cacheblend():
    pair = _pair()
    rng = random.Random()
    rng.random = lambda: 0.1  # type: ignore
    call = render_call(pair, rng=rng)  # cb is at A
    result = parse_response(call, "PREFER_A\nthe tool call is well-formed")
    assert result.label == "PREFER_A"
    assert result.prefers == "cacheblend"
    assert result.rationale == "the tool call is well-formed"


def test_parse_response_prefer_b_with_cb_at_a_resolves_no_cache():
    pair = _pair()
    rng = random.Random()
    rng.random = lambda: 0.1  # type: ignore
    call = render_call(pair, rng=rng)  # cb is at A
    result = parse_response(call, "PREFER_B\nB has a more sensible file path")
    assert result.label == "PREFER_B"
    assert result.prefers == "no_cache"


def test_parse_response_prefer_a_with_nc_at_a_resolves_no_cache():
    pair = _pair()
    rng = random.Random()
    rng.random = lambda: 0.9  # type: ignore
    call = render_call(pair, rng=rng)  # nc is at A
    result = parse_response(call, "PREFER_A\nA picks the right tool")
    assert result.prefers == "no_cache"


def test_parse_response_equivalent_resolves_equivalent_regardless_of_position():
    for pos in (0.1, 0.9):
        pair = _pair()
        rng = random.Random()
        rng.random = lambda: pos  # type: ignore
        call = render_call(pair, rng=rng)
        result = parse_response(call, "EQUIVALENT\nsame intent, cosmetic diff only")
        assert result.label == "EQUIVALENT"
        assert result.prefers == "equivalent"


def test_parse_response_unparseable_when_label_missing():
    pair = _pair()
    call = render_call(pair, rng=random.Random(0))
    result = parse_response(call, "I think A is better, but I'm not sure.")
    assert result.label == "UNPARSEABLE"
    assert result.prefers == "unparseable"


def test_parse_response_tolerates_leading_whitespace():
    pair = _pair()
    call = render_call(pair, rng=random.Random(0))
    result = parse_response(call, "   PREFER_A\nrationale here")
    assert result.label == "PREFER_A"


def test_parse_response_label_must_be_at_line_start():
    """A model that buries the label inside prose should be UNPARSEABLE."""
    pair = _pair()
    call = render_call(pair, rng=random.Random(0))
    result = parse_response(call, "I would say PREFER_A here")
    assert result.label == "UNPARSEABLE"
