"""Judge prompt template + response parsing.

Cacheblend's blended-KV outputs sometimes diverge from no_cache outputs at
T=0 (per §1 — 30% of substantive turns have <0.3 token identity). The
judge's job is pairwise preference: A vs B, with position randomization
on the call site (the prompt template itself is position-blind, so the
template can't leak which side is which).

Default judge: Sonnet-4.6 per design spec §3."""
from __future__ import annotations

import random
import re
from dataclasses import dataclass

JUDGE_SYSTEM = """\
You are evaluating two outputs from an AI coding agent. The agent is \
operating in an interactive multi-turn task where it uses tool calls (JSON \
function-call blocks) to read files, edit code, and make progress on \
software-engineering tasks.

Both outputs are responses from the SAME agent context to the SAME prompt — \
the only difference is the engine's KV-cache strategy. Your job is to \
identify which output is a better next step for the agent's task.

When comparing:
- A well-formed tool call (JSON specifying a function name + arguments) is \
generally better than natural-language prose, because the agent's task \
requires tool calls to make progress.
- When both outputs are tool calls, prefer the one with more appropriate \
arguments — sensible file paths, reasonable command flags, useful descriptions.
- When both outputs are prose, prefer the one that more directly addresses \
the task.
- If the outputs are functionally equivalent (e.g., same tool with cosmetically \
different argument wording), respond EQUIVALENT.

Respond with EXACTLY one of these tokens on the first line:
- PREFER_A
- PREFER_B
- EQUIVALENT

Then on a new line, give a 1-sentence rationale (≤ 30 words).
"""

JUDGE_USER_TEMPLATE = """\
TASK PROMPT (what the agent was asked to do):
{task_prompt}

OUTPUT A:
{output_a}

OUTPUT B:
{output_b}
"""


@dataclass
class JudgePair:
    capture_id: str
    turn_index: int
    cacheblend_text: str
    no_cache_text: str
    task_prompt: str


@dataclass
class JudgeCall:
    pair: JudgePair
    position_a: str  # "cacheblend" or "no_cache" — which side was rendered as A
    user_message: str  # rendered user message text


@dataclass
class JudgeResult:
    pair: JudgePair
    position_a: str
    raw_response: str
    label: str  # "PREFER_A" | "PREFER_B" | "EQUIVALENT" | "UNPARSEABLE"
    prefers: str  # "cacheblend" | "no_cache" | "equivalent" | "unparseable"
    rationale: str


def render_call(pair: JudgePair, *, rng: random.Random | None = None) -> JudgeCall:
    """Build a judge call with random A/B position assignment.

    `rng` is injectable for deterministic tests. Production callers pass a
    seeded rng per pair so the position assignment is recorded + reproducible.
    """
    rng = rng or random.Random()
    cb_is_a = rng.random() < 0.5
    a_text = pair.cacheblend_text if cb_is_a else pair.no_cache_text
    b_text = pair.no_cache_text if cb_is_a else pair.cacheblend_text
    user_message = JUDGE_USER_TEMPLATE.format(
        task_prompt=pair.task_prompt or "(none provided)",
        output_a=a_text or "(empty)",
        output_b=b_text or "(empty)",
    )
    return JudgeCall(
        pair=pair,
        position_a="cacheblend" if cb_is_a else "no_cache",
        user_message=user_message,
    )


_LABEL_RE = re.compile(r"^\s*(PREFER_A|PREFER_B|EQUIVALENT)\b", re.MULTILINE)


def parse_response(call: JudgeCall, raw: str) -> JudgeResult:
    """Map the judge's raw response to {prefers_cacheblend, prefers_no_cache,
    equivalent, unparseable}. position_a is used to translate PREFER_A/B back
    to cacheblend/no_cache."""
    m = _LABEL_RE.search(raw)
    label = m.group(1) if m else "UNPARSEABLE"
    if label == "EQUIVALENT":
        prefers = "equivalent"
    elif label == "PREFER_A":
        prefers = call.position_a
    elif label == "PREFER_B":
        prefers = "no_cache" if call.position_a == "cacheblend" else "cacheblend"
    else:
        prefers = "unparseable"
    # Rationale: everything after the label, trimmed
    rationale = ""
    if m:
        rationale = raw[m.end():].strip().split("\n", 1)[0].strip()
    return JudgeResult(
        pair=call.pair, position_a=call.position_a,
        raw_response=raw, label=label, prefers=prefers, rationale=rationale,
    )
