"""Tests for scripts/compaction_synth.py — the harness synthesis fallback.

The default litellm path requires a live backend; tests pass a mock
summarize_fn so the synthesis logic can be exercised without a network
call. Production usage runs through the litellm path."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from scripts.compaction_synth import (
    SUMMARIZER_SYSTEM_PROMPT,
    SUMMARY_BLOCK_TEMPLATE,
    build_postcompact_request,
    extract_system_prompt,
    reconstruct_conversation,
    synthesize,
)


# Schema mirrors TraceStore.SCHEMA but trimmed: this test doesn't need to
# round-trip the parquet token streams, only the request body JSONs.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ts_start REAL NOT NULL,
    ts_end REAL NOT NULL,
    prompt_token_count INTEGER NOT NULL,
    response_token_count INTEGER NOT NULL,
    cache_read_tokens INTEGER NOT NULL,
    cache_recompute_tokens INTEGER NOT NULL,
    engine_total_hit_tokens INTEGER,
    engine_load_tokens INTEGER,
    tokens_recomputed_hkvd INTEGER,
    chunk_aligned_hit_tokens INTEGER NOT NULL DEFAULT 0,
    invariant_violations TEXT NOT NULL DEFAULT '[]',
    ttft_ms REAL NOT NULL,
    request_body_json TEXT NOT NULL,
    token_parquet_path TEXT NOT NULL,
    lookups_parquet_path TEXT
);
"""


def _make_capture(dir_: Path, *, system_prompt: str, turns: list[tuple[str, str]]) -> Path:
    """Build a synthetic capture dir mimicking TraceStore output. Each
    cumulative request body grows the messages array — same shape produced
    by `claude --resume` against our proxy."""
    dir_.mkdir(parents=True, exist_ok=True)
    db_path = dir_ / "traces.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        for i, (role, content) in enumerate(turns):
            messages = [{"role": r, "content": c} for r, c in turns[: i + 1]]
            body = {
                "model": "claude-sonnet-4",
                "system": system_prompt,
                "messages": messages,
                "max_tokens": 1024,
            }
            conn.execute(
                """INSERT INTO requests VALUES
                   (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    f"req_{i}", "sess_a", float(i), float(i) + 0.5,
                    100, 50, 0, 0,
                    None, None, None, 0,
                    "[]", 5.0, json.dumps(body), "tokens/req_x.parquet", None,
                ),
            )
    return dir_


def test_reconstruct_conversation_yields_all_turns(tmp_path: Path):
    cap = _make_capture(
        tmp_path / "src",
        system_prompt="You are Claude.",
        turns=[
            ("user", "hi"),
            ("assistant", "hello"),
            ("user", "what is asyncio?"),
        ],
    )
    turns = reconstruct_conversation(cap)
    assert [t.role for t in turns] == ["user", "assistant", "user"]
    assert turns[2].content == "what is asyncio?"


def test_reconstruct_conversation_handles_block_content(tmp_path: Path):
    """Anthropic format wraps content in [{type: text, text: ...}] blocks.
    The reconstruct path must concatenate text blocks correctly."""
    cap = tmp_path / "src"
    cap.mkdir(parents=True, exist_ok=True)
    db_path = cap / "traces.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)
        body = {
            "model": "claude-sonnet-4",
            "system": "sys",
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "block one. "},
                    {"type": "text", "text": "block two."},
                ]},
            ],
        }
        conn.execute(
            """INSERT INTO requests VALUES
               (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "req_0", "sess_a", 0.0, 0.5, 10, 5, 0, 0,
                None, None, None, 0, "[]", 5.0, json.dumps(body),
                "tokens/x.parquet", None,
            ),
        )
    turns = reconstruct_conversation(cap)
    assert turns[0].content == "block one. block two."


def test_extract_system_prompt(tmp_path: Path):
    cap = _make_capture(
        tmp_path / "src",
        system_prompt="You are Claude. Your tools are Read, Edit.",
        turns=[("user", "ok")],
    )
    assert extract_system_prompt(cap).startswith("You are Claude.")


def test_build_postcompact_request_shape():
    body = build_postcompact_request(
        original_system="orig sys",
        summary_text="we discussed asyncio basics.",
        continuation_user_text="next step?",
        model="claude-sonnet-4",
    )
    # System has the original prompt + a recognizable summary block.
    assert body["system"].startswith("orig sys")
    assert "Conversation summary:" in body["system"]
    assert "we discussed asyncio basics." in body["system"]
    # Single continuation user message.
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"
    assert body["messages"][0]["content"] == "next step?"
    assert body["model"] == "claude-sonnet-4"


def test_summary_block_template_format():
    """Format must match what CC's natural autocompact emits — readers of
    fixtures (and the bench harness) key on the literal `Conversation
    summary:` prefix to find the block."""
    formatted = SUMMARY_BLOCK_TEMPLATE.format(summary_text="abc")
    assert "Conversation summary:" in formatted
    assert "abc" in formatted


def test_synthesize_writes_three_files(tmp_path: Path):
    """End-to-end with a mock summarize_fn — assert all three artifact
    files land in dest, with correct provenance."""
    src = _make_capture(
        tmp_path / "src",
        system_prompt="orig system prompt",
        turns=[
            ("user", "what is asyncio?"),
            ("assistant", "asyncio is python's concurrency lib."),
        ],
    )
    dest = tmp_path / "dest"

    captured: dict[str, list] = {"calls": []}
    def mock_summarize(turns):
        captured["calls"].append([(t.role, t.content) for t in turns])
        return "MOCK SUMMARY: discussed asyncio."

    out = synthesize(src, dest, summarize_fn=mock_summarize)
    assert out == dest

    body = json.loads((dest / "post_compact_request.json").read_text())
    summary = (dest / "summary.txt").read_text()
    meta = json.loads((dest / "meta.json").read_text())

    assert "MOCK SUMMARY" in summary
    assert "MOCK SUMMARY: discussed asyncio." in body["system"]
    assert body["system"].startswith("orig system prompt")
    assert meta["compaction_source"] == "synthetic_post_compact"
    assert meta["source_capture"] == str(src)
    assert meta["schema_version"] == "plan4_postcompact_v1"

    # The summarize_fn saw all the original turns.
    assert len(captured["calls"]) == 1
    assert captured["calls"][0] == [
        ("user", "what is asyncio?"),
        ("assistant", "asyncio is python's concurrency lib."),
    ]


def test_synthesize_requires_api_base_when_no_mock(tmp_path: Path):
    """Without a mock summarize_fn AND without summary_api_base/key, the
    function refuses to make an unconfigured network call."""
    src = _make_capture(
        tmp_path / "src",
        system_prompt="x",
        turns=[("user", "hi")],
    )
    with pytest.raises(ValueError, match="summarize_fn unset"):
        synthesize(src, tmp_path / "dest")


def test_summarizer_system_prompt_preserves_intent_keywords():
    """The summarizer system prompt must mention the things CC's
    autocompact actually preserves (goals, decisions, code identifiers,
    outstanding tasks) — otherwise the synthesized summary diverges in
    shape from natural autocompact."""
    for keyword in ("goals", "decisions", "file paths", "outstanding tasks"):
        assert keyword in SUMMARIZER_SYSTEM_PROMPT.lower(), keyword


def test_reconstruct_raises_on_missing_db(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        reconstruct_conversation(tmp_path / "nonexistent")
