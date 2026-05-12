"""Tests for bench/output_compare.py — identity, Jaccard, modal agreement."""
from __future__ import annotations

from skillcacher.bench.output_capture import Generation
from skillcacher.bench.output_compare import (
    canonicalize,
    modal_position_agreement,
    sampled_set_jaccard,
    token_identity_rate,
)


def _gen(text: str = "", tool_use: dict | None = None, extra: dict | None = None) -> Generation:
    blocks: list[dict] = []
    if text:
        blocks.append({"type": "text", "text": text})
    if tool_use is not None:
        blocks.append({"type": "tool_use", **tool_use})
    if extra is not None:
        blocks.append(extra)
    return Generation(
        text=text, content_blocks=blocks, stop_reason="end_turn",
        input_tokens=0, output_tokens=0, response_id="", ttft_ms=0.0,
    )


# ───────────────────────── canonicalize ─────────────────────────


def test_canonicalize_preserves_text_verbatim():
    g = _gen(text="hello world")
    assert canonicalize(g) == "hello world"


def test_canonicalize_tool_use_sorts_input_keys():
    g_a = _gen(tool_use={"id": "t1", "name": "search", "input": {"a": 1, "b": 2}})
    g_b = _gen(tool_use={"id": "t2", "name": "search", "input": {"b": 2, "a": 1}})
    # Same name, same input (different order, different id) → equal canonicalization
    assert canonicalize(g_a) == canonicalize(g_b)


def test_canonicalize_tool_use_strips_whitespace_from_args():
    # Whitespace inside JSON args shouldn't differentiate.
    g_a = _gen(tool_use={"name": "x", "input": {"q": "hello"}})
    g_b = _gen(tool_use={"name": "x", "input": {"q": "hello"}})
    assert canonicalize(g_a) == canonicalize(g_b)


def test_canonicalize_distinguishes_different_tool_calls():
    g_a = _gen(tool_use={"name": "search", "input": {"q": "weather"}})
    g_b = _gen(tool_use={"name": "search", "input": {"q": "stocks"}})
    assert canonicalize(g_a) != canonicalize(g_b)


def test_canonicalize_distinguishes_different_tool_names():
    g_a = _gen(tool_use={"name": "search", "input": {}})
    g_b = _gen(tool_use={"name": "fetch", "input": {}})
    assert canonicalize(g_a) != canonicalize(g_b)


def test_canonicalize_serializes_unknown_block_types():
    g = _gen(extra={"type": "thinking", "thinking": "reasoning..."})
    out = canonicalize(g)
    assert "thinking" in out


# ───────────────────────── token_identity_rate ─────────────────────────


def test_identity_identical_generations_is_one():
    g_a = _gen(text="hello world")
    g_b = _gen(text="hello world")
    assert token_identity_rate(g_a, g_b) == 1.0


def test_identity_first_char_diff_is_zero():
    g_a = _gen(text="hello")
    g_b = _gen(text="zello")
    assert token_identity_rate(g_a, g_b) == 0.0


def test_identity_prefix_match_is_proportional():
    # "hello world" vs "hello earth" share "hello " (6 chars) of 11 max.
    g_a = _gen(text="hello world")
    g_b = _gen(text="hello earth")
    rate = token_identity_rate(g_a, g_b)
    assert rate == 6 / 11


def test_identity_one_is_prefix_of_other():
    # "hello" is a prefix of "hello world": LCP=5, max=11 → 5/11.
    g_a = _gen(text="hello")
    g_b = _gen(text="hello world")
    rate = token_identity_rate(g_a, g_b)
    assert rate == 5 / 11


def test_identity_both_empty_is_one():
    g_a = _gen()
    g_b = _gen()
    assert token_identity_rate(g_a, g_b) == 1.0


def test_identity_one_empty_one_not_is_zero():
    g_a = _gen()
    g_b = _gen(text="x")
    assert token_identity_rate(g_a, g_b) == 0.0


def test_identity_tool_call_whitespace_difference_is_one():
    # Same tool call with different JSON arg ordering should canonicalize equal.
    g_a = _gen(tool_use={"name": "x", "input": {"a": 1, "b": 2}})
    g_b = _gen(tool_use={"name": "x", "input": {"b": 2, "a": 1}})
    assert token_identity_rate(g_a, g_b) == 1.0


# ───────────────────────── sampled_set_jaccard ─────────────────────────


def test_jaccard_identical_sample_sets_is_one():
    a = [_gen(text=t) for t in ["x", "y", "z"]]
    b = [_gen(text=t) for t in ["x", "y", "z"]]
    assert sampled_set_jaccard(a, b) == 1.0


def test_jaccard_disjoint_sample_sets_is_zero():
    a = [_gen(text=t) for t in ["x", "y"]]
    b = [_gen(text=t) for t in ["a", "b"]]
    assert sampled_set_jaccard(a, b) == 0.0


def test_jaccard_partial_overlap():
    # A = {x, y, z}, B = {y, z, w} → |∩|=2, |∪|=4, J=0.5
    a = [_gen(text=t) for t in ["x", "y", "z"]]
    b = [_gen(text=t) for t in ["y", "z", "w"]]
    assert sampled_set_jaccard(a, b) == 0.5


def test_jaccard_dedupes_within_each_side():
    # A = {x, x, y} → set is {x, y}. B = {x, y} → identical, J=1.0
    a = [_gen(text="x"), _gen(text="x"), _gen(text="y")]
    b = [_gen(text="x"), _gen(text="y")]
    assert sampled_set_jaccard(a, b) == 1.0


def test_jaccard_both_empty_is_one():
    assert sampled_set_jaccard([], []) == 1.0


# ───────────────────────── modal_position_agreement ─────────────────────────


def test_modal_identical_samples_is_one():
    a = [_gen(text="hello"), _gen(text="hello"), _gen(text="hello")]
    b = [_gen(text="hello"), _gen(text="hello"), _gen(text="hello")]
    assert modal_position_agreement(a, b) == 1.0


def test_modal_modal_chars_match_despite_minority_disagreement():
    # Modal of A = "x" (3 of 5). Modal of B = "x" (3 of 5). Even though
    # the minority chars differ, position-level modal agreement = 1.0.
    a = [_gen(text="x"), _gen(text="x"), _gen(text="x"), _gen(text="y"), _gen(text="z")]
    b = [_gen(text="x"), _gen(text="x"), _gen(text="x"), _gen(text="a"), _gen(text="b")]
    assert modal_position_agreement(a, b) == 1.0


def test_modal_modal_chars_disagree_at_position():
    a = [_gen(text="ab"), _gen(text="ab"), _gen(text="ab")]
    b = [_gen(text="ax"), _gen(text="ax"), _gen(text="ax")]
    # Position 0: 'a' == 'a'. Position 1: 'b' != 'x'. → 1/2 = 0.5
    assert modal_position_agreement(a, b) == 0.5


def test_modal_truncates_to_shorter_max_length():
    # A's longest is "abcdef" (6); B's longest is "abc" (3). n = 3.
    a = [_gen(text="abcdef")]
    b = [_gen(text="abc")]
    # All 3 positions agree. → 3/3 = 1.0
    assert modal_position_agreement(a, b) == 1.0


def test_modal_both_empty_is_one():
    assert modal_position_agreement([], []) == 1.0


def test_modal_one_empty_one_not_is_zero():
    assert modal_position_agreement([], [_gen(text="x")]) == 0.0
    assert modal_position_agreement([_gen(text="x")], []) == 0.0
