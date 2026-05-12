"""CC-aware segmentation parser.

Splits a Claude Code prompt body's text content at recognized structural
boundaries and injects ` # # ` separator tokens between blocks. Without
this, cacheblend's segment detector finds 0 segments on natural CC traffic
(prior development spike, 48/48 LMCache lines reported
`(computed=0, hit=0, need_to_load=0)` against the asyncio Q&A capture).

The parser is content-stable across turns: it keys on STRUCTURAL anchors
(literal substrings like `Status:\n`, `Recent commits:\n`,
`<system-reminder>`, `Summary:\n1. `, etc.) rather than on prompt content
or CC-version strings. It never edits the captured content; it only
injects separators between detected blocks.

Wired into proxy/server.py at request entry (before assemble_and_tokenize
and before forwarding to the backend) so both the proxy's tagging path
and the backend's lmcache chunk detector see the rewritten body.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

# Default separator. Mirrors the LMCACHE_BLEND_SPECIAL_STR set in
# scripts/dev/oneshot_pod.py for the cacheblend condition. Surrounded by
# spaces so BPE merges keep the two `#` tokens distinct (matches the harness's
# canonical-bench tokenization on Llama-3 tokenizer).
DEFAULT_SEPARATOR = " # # "

# A request is recognized as CC-shaped if any of its text payloads contains
# this anchor. Stable across CC versions back to ~2.0; the version digits
# after `cc_version=` change but the field name doesn't.
CC_ANCHOR = re.compile(r"x-anthropic-billing-header:\s*cc_version=")

# fix (): the cc_header contains two fields
# that vary per-turn and per-session — `cch=<hash>` (CC's per-request cache
# fingerprint) and `cc_version=2.1.X.<build_suffix>` (build suffix differs
# between regular and /compact-internal turns). Together these make the
# first chunk's hash unique per turn → cacheblend can never hit on chunk 0
# → lookup returns 0 immediately (prefix-match short-circuits on first
# miss). We normalize these to stable placeholders before injecting
# separators so chunk 0 hashes identically across turns of the same session.
# The backend (vllm/Llama) doesn't parse this header — it's CC-side billing
# metadata stuffed into the prompt — so normalization is safe.
_CCH_VAR = re.compile(r"\bcch=[A-Za-z0-9]+;")
_CC_VERSION_VAR = re.compile(r"\bcc_version=\d+\.\d+\.\d+\.[A-Za-z0-9]+;")


def _normalize_cc_header(text: str) -> str:
    """Replace per-turn / per-build cc_header values with stable placeholders.
    Idempotent: re-running on already-normalized text is a no-op."""
    text = _CCH_VAR.sub("cch=NORM;", text)
    text = _CC_VERSION_VAR.sub("cc_version=NORM;", text)
    return text


@dataclass(frozen=True)
class Segment:
    """A recognized structural block in a CC prompt text payload."""
    start: int
    end: int
    kind: str  # one of: cc_header, gitStatus, recent_commits,
               # system_reminder, compaction_preamble, summary,
               # jsonl_backref, command_block

    def slice(self, text: str) -> str:
        return text[self.start:self.end]


# ---------------------------------------------------------------------------
# Structural anchors. Each pattern has a unique kind tag and a precedence
# (tiebreak when matches overlap). Higher precedence wins.
# ---------------------------------------------------------------------------

# `<system-reminder>...\n</system-reminder>`. Multiline, non-greedy.
_SYSTEM_REMINDER = re.compile(
    r"<system-reminder>\n.*?\n</system-reminder>",
    re.DOTALL,
)

# `<(command-name|local-command-stdout|local-command-caveat|command-message|command-args)>...</...>`.
# Self-closing variants too (`<command-args></command-args>` is empty content).
_COMMAND_BLOCK = re.compile(
    r"<(command-name|local-command-stdout|local-command-caveat"
    r"|command-message|command-args)>.*?</\1>",
    re.DOTALL,
)

# Compaction continuation preamble. Stable wording from CC's autocompact
# prelude (verified).
_COMPACTION_PREAMBLE = re.compile(
    r"This session is being continued from a previous conversation "
    r"that ran out of context\."
    r".*?"
    r"(?=\n\nSummary:\n|\Z)",
    re.DOTALL,
)

# 9-section CC compaction summary. Starts at `Summary:\n1. ` and runs
# until the JSONL backreference (`If you need specific details from
# before compaction`) or end-of-text.
_SUMMARY_BLOCK = re.compile(
    r"Summary:\n1\.\s.*?"
    r"(?=\nIf you need specific details from before compaction|\Z)",
    re.DOTALL,
)

# JSONL backreference paragraph. Starts at `If you need specific details
# from before compaction` and runs through the `.jsonl` filename.
_JSONL_BACKREF = re.compile(
    r"If you need specific details from before compaction.*?\.jsonl",
    re.DOTALL,
)

# CC system header — starts at `x-anthropic-billing-header:` and runs
# through CWD + Date lines. Bounded by either the next blank line or the
# `gitStatus:` marker.
_CC_HEADER = re.compile(
    r"x-anthropic-billing-header:.*?Date:[^\n]*",
    re.DOTALL,
)

# gitStatus preamble + Status block + Recent commits block, each as their
# own segment. Anchored on the literal heading lines.
_GIT_STATUS_BLOCK = re.compile(
    r"gitStatus:.*?(?=\nRecent commits:\n|\Z)",
    re.DOTALL,
)
_RECENT_COMMITS_BLOCK = re.compile(
    r"Recent commits:\n.*?(?=\n\n|\Z)",
    re.DOTALL,
)


def _all_segments(text: str) -> list[Segment]:
    """Find every structural span in `text`, in document order, with
    overlapping spans resolved by precedence: command_block >
    system_reminder > summary > jsonl_backref > compaction_preamble >
    cc_header > recent_commits > gitStatus."""
    raw: list[tuple[int, int, str, int]] = []  # (start, end, kind, prec)
    PATTERNS = [
        (_COMMAND_BLOCK,         "command_block",        80),
        (_SYSTEM_REMINDER,       "system_reminder",      70),
        (_SUMMARY_BLOCK,         "summary",              60),
        (_JSONL_BACKREF,         "jsonl_backref",        50),
        (_COMPACTION_PREAMBLE,   "compaction_preamble",  40),
        (_CC_HEADER,             "cc_header",            30),
        (_RECENT_COMMITS_BLOCK,  "recent_commits",       20),
        (_GIT_STATUS_BLOCK,      "gitStatus",            10),
    ]
    for pat, kind, prec in PATTERNS:
        for m in pat.finditer(text):
            raw.append((m.start(), m.end(), kind, prec))
    if not raw:
        return []
    # Sort by start; break overlapping pairs by keeping higher precedence.
    raw.sort(key=lambda r: (r[0], -r[3]))
    accepted: list[tuple[int, int, str, int]] = []
    for r in raw:
        if accepted and r[0] < accepted[-1][1]:
            # Overlap with previous accepted span. Keep whichever has
            # higher precedence; if tied, keep the longer span.
            prev = accepted[-1]
            if r[3] > prev[3] or (r[3] == prev[3] and (r[1] - r[0]) > (prev[1] - prev[0])):
                accepted[-1] = r
            # else drop r
            continue
        accepted.append(r)
    return [Segment(start=s, end=e, kind=k) for (s, e, k, _) in accepted]


def find_segments(text: str) -> list[Segment]:
    """Public entry point — return the segment list for a single text
    payload, in document order, with no overlaps."""
    return _all_segments(text)


def inject_separators(text: str, *, separator: str | None = None) -> str:
    """Walk `text`, find structural segments, and emit `text` with
    `separator` inserted at every segment boundary (both before and after
    each segment, but never duplicated when adjacent boundaries collide).

    Idempotent: the separator string itself is never targeted by any
    pattern, so re-running on already-separated text is a no-op apart
    from collapsing duplicate adjacent separators.

    Per §1.6: also normalizes the per-turn `cch=` and `cc_version=`
    placeholders inside the cc_header so chunk 0 hashes identically
    across turns of the same session (otherwise cacheblend's prefix-match
    lookup misses on chunk 0 and returns 0 immediately)."""
    if not text:
        return text
    sep = separator if separator is not None else DEFAULT_SEPARATOR
    text = _normalize_cc_header(text)
    segments = find_segments(text)
    if not segments:
        return text

    pieces: list[str] = []
    cursor = 0
    for seg in segments:
        # Inter-segment text first (between previous boundary and this start).
        if seg.start > cursor:
            pieces.append(text[cursor:seg.start])
        pieces.append(sep)
        pieces.append(text[seg.start:seg.end])
        pieces.append(sep)
        cursor = seg.end
    # Trailing text after the last segment.
    if cursor < len(text):
        pieces.append(text[cursor:])
    out = "".join(pieces)
    # Collapse any adjacent duplicate separators that arise when two
    # segments are exactly back-to-back.
    duplicate = sep + sep
    while duplicate in out:
        out = out.replace(duplicate, sep)
    return out


# ---------------------------------------------------------------------------
# Request body rewriting (the public API the proxy calls).
# ---------------------------------------------------------------------------


def is_cc_request(body: dict[str, Any]) -> bool:
    """Heuristic: does this Anthropic-Messages request body look like one
    that CC sent? Keys on the `cc_version=` substring in any text payload."""
    sys_b = body.get("system")
    if isinstance(sys_b, str) and CC_ANCHOR.search(sys_b):
        return True
    if isinstance(sys_b, list):
        for b in sys_b:
            if b.get("type") == "text" and CC_ANCHOR.search(b.get("text", "")):
                return True
    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str) and CC_ANCHOR.search(c):
            return True
        if isinstance(c, list):
            for blk in c:
                if blk.get("type") == "text" and CC_ANCHOR.search(blk.get("text", "")):
                    return True
    return False


def rewrite_request_body(
    body: dict[str, Any], *, separator: str | None = None,
) -> dict[str, Any]:
    """Return a new request body with ` # # ` separators injected around
    structural blocks in every text payload. If `body` doesn't look CC-
    shaped, it's returned unchanged.

    Non-text payloads (tool_use/tool_result blocks, tool definitions) are
    left untouched — those have their own structural boundaries that the
    span tagger handles separately."""
    if not is_cc_request(body):
        return body
    sep = separator if separator is not None else DEFAULT_SEPARATOR
    out: dict[str, Any] = dict(body)

    sys_b = body.get("system")
    if isinstance(sys_b, str):
        out["system"] = inject_separators(sys_b, separator=sep)
    elif isinstance(sys_b, list):
        out["system"] = [
            ({**b, "text": inject_separators(b.get("text", ""), separator=sep)}
             if b.get("type") == "text" else b)
            for b in sys_b
        ]

    new_messages: list[dict[str, Any]] = []
    for m in body.get("messages") or []:
        nm: dict[str, Any] = dict(m)
        c = m.get("content")
        if isinstance(c, str):
            nm["content"] = inject_separators(c, separator=sep)
        elif isinstance(c, list):
            nm["content"] = [
                ({**blk, "text": inject_separators(blk.get("text", ""), separator=sep)}
                 if blk.get("type") == "text" else blk)
                for blk in c
            ]
        new_messages.append(nm)
    out["messages"] = new_messages
    return out


def is_enabled() -> bool:
    """Feature-flag the parser. Default: enabled. Set
    SKILLCACHER_CC_SEGMENT_PARSER=false to disable (e.g. for A/B comparing
    the capture pre-parser baseline against post-parser numbers)."""
    return os.environ.get("SKILLCACHER_CC_SEGMENT_PARSER", "true").strip().lower() not in (
        "false", "0", "no", "off",
    )
