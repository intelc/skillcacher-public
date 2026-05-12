"""CC-aware segmentation parser tests.

Anchor matching, boundary preservation, idempotence, and a roundtrip
against the actual post-compact request body captured during the
2026-05-08 capture (tests/fixtures/.../plan4_compaction_spike/)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from skillcacher.proxy.cc_segment_parser import (
    CC_ANCHOR,
    DEFAULT_SEPARATOR,
    Segment,
    _normalize_cc_header,
    find_segments,
    inject_separators,
    is_cc_request,
    is_enabled,
    rewrite_request_body,
)


# ---------------------------------------------------------------------------
# Anchor / segment detection
# ---------------------------------------------------------------------------

def test_finds_system_reminder_block():
    text = (
        "lead-in text\n"
        "<system-reminder>\n"
        "context note here\n"
        "</system-reminder>\n"
        "trailing text"
    )
    segs = find_segments(text)
    sr = [s for s in segs if s.kind == "system_reminder"]
    assert len(sr) == 1
    assert sr[0].slice(text).startswith("<system-reminder>")
    assert sr[0].slice(text).endswith("</system-reminder>")


def test_finds_command_block():
    text = "preamble <command-name>/compact</command-name> postamble"
    segs = find_segments(text)
    cb = [s for s in segs if s.kind == "command_block"]
    assert len(cb) == 1
    assert cb[0].slice(text) == "<command-name>/compact</command-name>"


def test_finds_summary_block_with_9_sections():
    text = (
        "Summary:\n"
        "1. Primary Request: blah\n\n"
        "2. Key Concepts: blah\n\n"
        "3. Files: blah\n\n"
        "4. Errors: blah\n\n"
        "5. Problem Solving: blah\n\n"
        "6. All user messages: blah\n\n"
        "7. Pending Tasks: blah\n\n"
        "8. Current Work: blah\n\n"
        "9. Optional Next Step: blah\n\n"
        "If you need specific details from before compaction, see /tmp/x.jsonl"
    )
    segs = find_segments(text)
    summary = [s for s in segs if s.kind == "summary"]
    assert len(summary) == 1
    body = summary[0].slice(text)
    assert body.startswith("Summary:\n1.")
    assert "9. Optional Next Step" in body
    # Backreference is its own segment, not folded into the summary.
    assert "If you need specific details" not in body


def test_finds_jsonl_backref():
    text = (
        "blah\n"
        "If you need specific details from before compaction (like exact code), "
        "read the full transcript at: /Users/x/.claude/projects/abc.jsonl\n"
        "more text"
    )
    segs = find_segments(text)
    br = [s for s in segs if s.kind == "jsonl_backref"]
    assert len(br) == 1
    assert br[0].slice(text).endswith(".jsonl")


def test_finds_compaction_preamble():
    text = (
        "<system-reminder>\nctx\n</system-reminder>\n\n"
        "This session is being continued from a previous conversation "
        "that ran out of context. The summary below covers the earlier portion.\n\n"
        "Summary:\n1. Primary Request: x"
    )
    segs = find_segments(text)
    pre = [s for s in segs if s.kind == "compaction_preamble"]
    assert len(pre) == 1
    assert pre[0].slice(text).startswith("This session is being continued")
    # Stops before "Summary:".
    assert "Summary:" not in pre[0].slice(text)


def test_finds_cc_header_through_date():
    text = (
        "x-anthropic-billing-header: cc_version=2.1.136.829; cc_entrypoint=sdk-cli; cch=abc;"
        "You are a Claude agent, built on Anthropic's Claude Agent SDK."
        "CWD: /Users/x/codestuff\n"
        "Date: 2026-05-08\n"
        "\n"
        "gitStatus: This is the git status..."
    )
    segs = find_segments(text)
    h = [s for s in segs if s.kind == "cc_header"]
    assert len(h) == 1
    body = h[0].slice(text)
    assert body.startswith("x-anthropic-billing-header:")
    assert "Date: 2026-05-08" in body
    # gitStatus belongs to the next segment, not this one.
    assert "gitStatus:" not in body


def test_finds_recent_commits_block():
    text = (
        "gitStatus: ...\n"
        "Status:\nM file.py\n"
        "Recent commits:\n"
        "abc123 first\n"
        "def456 second\n"
        "\n"
        "trailing"
    )
    segs = find_segments(text)
    rc = [s for s in segs if s.kind == "recent_commits"]
    assert len(rc) == 1
    body = rc[0].slice(text)
    assert body.startswith("Recent commits:")
    assert "def456 second" in body
    assert "trailing" not in body


def test_overlapping_summary_inside_system_reminder_resolves_by_precedence():
    """A `<system-reminder>` that happens to enclose a `Summary:\\n1. `
    line should be matched as system_reminder (precedence 70) rather
    than summary (60). Without the precedence rule, both would match
    overlappingly."""
    text = (
        "<system-reminder>\n"
        "Summary:\n1. nested item — should NOT be its own segment\n"
        "</system-reminder>"
    )
    segs = find_segments(text)
    kinds = [s.kind for s in segs]
    assert "system_reminder" in kinds
    assert "summary" not in kinds


# ---------------------------------------------------------------------------
# Separator injection
# ---------------------------------------------------------------------------

def test_inject_separators_no_segments_passes_through():
    text = "just plain text with no CC structural anchors at all"
    assert inject_separators(text) == text


def test_inject_separators_wraps_each_segment():
    text = "lead <system-reminder>\nfoo\n</system-reminder> tail"
    out = inject_separators(text)
    # Separator before AND after the matched block.
    assert out.count(DEFAULT_SEPARATOR) == 2
    # Original content survives in the output (modulo separators).
    assert "<system-reminder>\nfoo\n</system-reminder>" in out
    # Lead/tail text preserved.
    assert "lead" in out
    assert "tail" in out


def test_inject_separators_collapses_duplicate_separators_at_boundaries():
    """When two segments are back-to-back, the trailing separator of one
    and the leading separator of the next collapse into a single one."""
    text = (
        "<system-reminder>\nA\n</system-reminder>"
        "<command-name>/x</command-name>"
    )
    out = inject_separators(text)
    # Should be: SEP <sr> SEP <cmd> SEP — three, not four.
    assert out.count(DEFAULT_SEPARATOR) == 3
    assert DEFAULT_SEPARATOR + DEFAULT_SEPARATOR not in out


def test_inject_separators_idempotent():
    text = "lead <system-reminder>\nfoo\n</system-reminder> tail"
    once = inject_separators(text)
    twice = inject_separators(once)
    assert once == twice


def test_inject_separators_preserves_non_normalized_content_byte_for_byte():
    """inject_separators normalizes per-turn cch= and
    cc_version= placeholders. Stripping separators yields the
    POST-NORMALIZATION text, which differs from the input only in those
    two fields. All other content survives byte-for-byte."""
    text = (
        "x-anthropic-billing-header: cc_version=2.1.136.829; cch=abcdef;"
        "CWD: /repo\nDate: 2026-05-08\n\n"
        "gitStatus: blah\nRecent commits:\nabc first\n\n"
        "user follow-up text"
    )
    out = inject_separators(text)
    stripped = out.replace(DEFAULT_SEPARATOR, "")
    # Non-cch/cc_version content survives.
    assert "CWD: /repo" in stripped
    assert "Date: 2026-05-08" in stripped
    assert "gitStatus: blah" in stripped
    assert "Recent commits:" in stripped
    assert "abc first" in stripped
    assert "user follow-up text" in stripped
    # Per-turn varying fields normalized.
    assert "cc_version=NORM;" in stripped
    assert "cch=NORM;" in stripped
    # Non-varying field preserved.
    assert "x-anthropic-billing-header:" in stripped


# ---------------------------------------------------------------------------
# Request-body rewriting (the proxy entry point)
# ---------------------------------------------------------------------------

def test_is_cc_request_keys_on_billing_header_anchor():
    cc = {"system": "x-anthropic-billing-header: cc_version=2.1.136.829;"}
    not_cc = {"system": "You are a helpful assistant."}
    assert is_cc_request(cc) is True
    assert is_cc_request(not_cc) is False


def test_is_cc_request_works_in_messages_array():
    body = {
        "system": "boring",
        "messages": [
            {"role": "user", "content": "x-anthropic-billing-header: cc_version=2.1.137.999;"},
        ],
    }
    assert is_cc_request(body) is True


def test_rewrite_request_body_passes_through_non_cc():
    body = {"system": "You are a helpful assistant.",
            "messages": [{"role": "user", "content": "hello"}]}
    out = rewrite_request_body(body)
    assert out is body  # identity — no rewrite path executed


def test_rewrite_request_body_injects_in_system_and_messages():
    body = {
        "system": (
            "x-anthropic-billing-header: cc_version=2.1.136.829;"
            "You are a Claude agent.CWD: /repo\nDate: 2026-05-08"
        ),
        "messages": [
            {"role": "user", "content": "<system-reminder>\nhi\n</system-reminder>"},
            {"role": "assistant", "content": "ok"},
        ],
    }
    out = rewrite_request_body(body)
    assert DEFAULT_SEPARATOR in out["system"]
    # The user message has a system-reminder block — gets separators.
    assert DEFAULT_SEPARATOR in out["messages"][0]["content"]
    # Assistant "ok" has no segments, passes through.
    assert out["messages"][1]["content"] == "ok"
    # Original body untouched (defensive copy).
    assert "system-reminder" in body["messages"][0]["content"]
    assert DEFAULT_SEPARATOR not in body["system"]


def test_rewrite_request_body_handles_block_content_form():
    body = {
        "system": [
            {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.136.829;"
                                    "CWD: /r\nDate: 2026-05-08"},
        ],
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": "<system-reminder>\nctx\n</system-reminder>"},
            ]},
        ],
    }
    out = rewrite_request_body(body)
    assert DEFAULT_SEPARATOR in out["system"][0]["text"]
    assert DEFAULT_SEPARATOR in out["messages"][0]["content"][0]["text"]


def test_is_enabled_default_on(monkeypatch):
    monkeypatch.delenv("SKILLCACHER_CC_SEGMENT_PARSER", raising=False)
    assert is_enabled() is True


def test_is_enabled_respects_off_flag(monkeypatch):
    for off in ("false", "0", "no", "off", "FALSE"):
        monkeypatch.setenv("SKILLCACHER_CC_SEGMENT_PARSER", off)
        assert is_enabled() is False


# ---------------------------------------------------------------------------
# Roundtrip on the actual capture fixture
# ---------------------------------------------------------------------------

FIXTURE = (
    Path(__file__).parent.parent / "fixtures" / "claude_code_real" / "post_compact"
    / "plan4_compaction_spike"
)


def test_normalize_cc_header_strips_cch_and_version():
    """per-turn `cch=...` and per-build `cc_version=...` make
    chunk 0 hash differently across turns of the same session — cacheblend
    prefix-match misses immediately. _normalize_cc_header swaps both to
    stable placeholders so chunk 0 stabilizes."""
    turn1 = (
        "x-anthropic-billing-header: cc_version=2.1.137.e1b; cc_entrypoint=sdk-cli; "
        "cch=7381a;You are a Claude agent."
    )
    turn2 = (
        "x-anthropic-billing-header: cc_version=2.1.137.27b; cc_entrypoint=sdk-cli; "
        "cch=a0190;You are a Claude agent."
    )
    n1 = _normalize_cc_header(turn1)
    n2 = _normalize_cc_header(turn2)
    assert n1 == n2, f"normalized headers must be identical:\n  {n1!r}\n  {n2!r}"
    assert "cc_version=NORM;" in n1
    assert "cch=NORM;" in n1
    # The non-varying parts must survive untouched.
    assert "cc_entrypoint=sdk-cli" in n1
    assert "You are a Claude agent." in n1


def test_normalize_cc_header_idempotent():
    text = "x-anthropic-billing-header: cc_version=NORM; cc_entrypoint=sdk-cli; cch=NORM;You are a Claude agent."
    assert _normalize_cc_header(text) == text


def test_inject_separators_normalizes_cc_header_first():
    """Verify the full inject_separators pipeline applies normalization
    before splitting at structural anchors (so the cc_header segment's
    content is stable across turns)."""
    turn1 = (
        "x-anthropic-billing-header: cc_version=2.1.137.e1b; cc_entrypoint=sdk-cli; "
        "cch=7381a;You are a Claude agent.CWD: /repo\nDate: 2026-05-09\n\n"
        "gitStatus: blah"
    )
    turn2 = (
        "x-anthropic-billing-header: cc_version=2.1.137.27b; cc_entrypoint=sdk-cli; "
        "cch=a0190;You are a Claude agent.CWD: /repo\nDate: 2026-05-09\n\n"
        "gitStatus: blah"
    )
    out1 = inject_separators(turn1)
    out2 = inject_separators(turn2)
    assert out1 == out2, "post-normalization separator-injected output must be identical across turns"


@pytest.mark.skipif(
    not (FIXTURE / "traces.sqlite").exists(),
    reason="capture fixture not present (gitignored .sqlite)",
)
def test_roundtrip_on_real_post_compact_request():
    """Load the actual post-compact request body from the capture, run
    the parser, verify ` # # ` separators land at expected boundaries
    AND non-volatile content survives. Per-turn cch= and cc_version=
    placeholders are normalized to NORM (§1.6 fix)."""
    with sqlite3.connect(FIXTURE / "traces.sqlite") as con:
        con.row_factory = sqlite3.Row
        rows = list(con.execute(
            "SELECT request_body_json FROM requests ORDER BY ts_start"
        ))
    # The post-compact turn was at index 13 in the capture.
    body = json.loads(rows[13]["request_body_json"])
    out = rewrite_request_body(body)

    def _join_sys(s):
        if isinstance(s, str):
            return s
        return "".join(b.get("text", "") for b in s if b.get("type") == "text")
    sys_joined = _join_sys(out["system"])
    sys_stripped = sys_joined.replace(DEFAULT_SEPARATOR, "")
    # Non-volatile content survives (CWD, Date, gitStatus markers).
    assert "CWD:" in sys_stripped
    assert "Date:" in sys_stripped
    # Volatile fields normalized.
    assert "cc_version=NORM;" in sys_stripped
    assert "cch=NORM;" in sys_stripped

    # The post-compact msg[0] has the Summary + jsonl_backref + system_reminder
    # blocks — at least 4 separator pairs land.
    m0_orig = body["messages"][0]["content"]
    if isinstance(m0_orig, list):
        m0_orig = "".join(b.get("text", "") for b in m0_orig if b.get("type") == "text")
    m0_out = out["messages"][0]["content"]
    if isinstance(m0_out, list):
        m0_out = "".join(b.get("text", "") for b in m0_out if b.get("type") == "text")

    sep_count = m0_out.count(DEFAULT_SEPARATOR)
    assert sep_count >= 4, f"expected ≥4 separators in post-compact msg[0]; got {sep_count}"
    # msg[0] doesn't contain cc_version/cch fields, so byte-preservation holds there.
    assert m0_out.replace(DEFAULT_SEPARATOR, "") == m0_orig

    # The Summary block is one of the matched segments (post-compact is
    # all about the summary).
    segs = find_segments(m0_orig)
    kinds = {s.kind for s in segs}
    assert "summary" in kinds, f"expected summary segment in post-compact msg[0]; kinds={kinds}"
