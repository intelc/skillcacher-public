"""Tests for judge/sonnet_judge.py — verifies the SDK call shape and the
parse roundtrip via a stubbed Anthropic client. No network."""
from __future__ import annotations

import random
from dataclasses import dataclass

import pytest

from skillcacher.judge.prompt import JUDGE_SYSTEM, JudgePair, render_call
from skillcacher.judge.sonnet_judge import call_judge, call_judge_with_retry


@dataclass
class _StubBlock:
    type: str
    text: str


@dataclass
class _StubResponse:
    content: list


class _StubClient:
    def __init__(self, response_text: str = "PREFER_A\nrationale"):
        self.response_text = response_text
        self.last_kwargs: dict | None = None

    @property
    def messages(self):
        return self

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _StubResponse(content=[_StubBlock(type="text", text=self.response_text)])


def _pair() -> JudgePair:
    return JudgePair(capture_id="cap", turn_index=0,
                     cacheblend_text="CB", no_cache_text="NC", task_prompt="P")


def test_call_judge_sends_system_with_cache_control():
    client = _StubClient("PREFER_A\nx")
    call = render_call(_pair(), rng=random.Random(0))
    call_judge(call, client=client, model="claude-sonnet-4-6")
    kw = client.last_kwargs
    assert kw["model"] == "claude-sonnet-4-6"
    assert kw["max_tokens"] == 200
    assert isinstance(kw["system"], list)
    assert kw["system"][0]["text"] == JUDGE_SYSTEM
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    # User message is rendered from the call
    assert kw["messages"][0]["role"] == "user"
    assert "CB" in kw["messages"][0]["content"]
    assert "NC" in kw["messages"][0]["content"]


def test_call_judge_returns_parsed_result():
    client = _StubClient("PREFER_A\nbecause A wins")
    rng = random.Random()
    rng.random = lambda: 0.1  # type: ignore
    call = render_call(_pair(), rng=rng)  # cb at A
    result = call_judge(call, client=client)
    assert result.label == "PREFER_A"
    assert result.prefers == "cacheblend"
    assert result.rationale == "because A wins"


def test_call_judge_concatenates_multi_block_text_response():
    @dataclass
    class _MultiBlockResponse:
        content: list

    class _MultiClient:
        def __init__(self):
            self.last_kwargs = None
        @property
        def messages(self): return self
        def create(self, **kwargs):
            self.last_kwargs = kwargs
            return _MultiBlockResponse(content=[
                _StubBlock(type="text", text="PREFER_B\n"),
                _StubBlock(type="text", text="reasoning continues"),
            ])

    rng = random.Random()
    rng.random = lambda: 0.1
    call = render_call(_pair(), rng=rng)
    result = call_judge(call, client=_MultiClient())
    assert result.label == "PREFER_B"
    assert result.prefers == "no_cache"


def test_call_judge_with_retry_returns_first_parseable():
    client = _StubClient("PREFER_A\nok")
    call = render_call(_pair(), rng=random.Random(0))
    result = call_judge_with_retry(call, client=client, max_retries=3)
    assert result.label == "PREFER_A"


def test_call_judge_with_retry_retries_on_unparseable_then_returns_last():
    """If every attempt is unparseable, return the last result rather than raising."""
    class _AlwaysUnparseable:
        def __init__(self): self.calls = 0
        @property
        def messages(self): return self
        def create(self, **kwargs):
            self.calls += 1
            return _StubResponse(content=[_StubBlock(type="text", text="Hmm I am not sure")])
    client = _AlwaysUnparseable()
    call = render_call(_pair(), rng=random.Random(0))
    result = call_judge_with_retry(call, client=client, max_retries=3)
    assert result.label == "UNPARSEABLE"
    assert client.calls == 3
