"""Render an Anthropic Messages request into a prompt-token sequence and
emit a per-token tag list. Uses the served model's HuggingFace tokenizer.

Tag kinds align with the structural span tagger:
  system_static | tool_def | skill_body | dynamic | other
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

from skillcacher.tagging.span_tagger import (
    SKILL_FRONTMATTER_RE,
    DYNAMIC_BLOCK_RE,
    _tokenizer,
)

if TYPE_CHECKING:
    from skillcacher.tagging.skill_prefix_index import SkillPrefixIndex


def _encode(text: str, name: str) -> list[int]:
    tok = _tokenizer(name)
    return tok.encode(text, add_special_tokens=False)


def _emit_with_tag(text: str, tag: str, name: str, out_tokens: list[int], out_tags: list[str]) -> None:
    if not text:
        return
    ids = _encode(text, name)
    out_tokens.extend(ids)
    out_tags.extend([tag] * len(ids))


def _emit_text_block_split_dynamic(text: str, base_tag: str, name: str,
                                    out_tokens: list[int], out_tags: list[str]) -> None:
    """Split a text block on !`cmd` markers, tagging the markers as dynamic."""
    cursor = 0
    for m in DYNAMIC_BLOCK_RE.finditer(text):
        if m.start() > cursor:
            _emit_with_tag(text[cursor:m.start()], base_tag, name, out_tokens, out_tags)
        _emit_with_tag(m.group(0), "dynamic", name, out_tokens, out_tags)
        cursor = m.end()
    if cursor < len(text):
        _emit_with_tag(text[cursor:], base_tag, name, out_tokens, out_tags)


def assemble_and_tokenize(body: dict, tokenizer_name: str) -> tuple[list[int], list[str]]:
    """Walk the Anthropic request and emit (token_ids, tags) parallel arrays.

    The exact prompt bytes here will not byte-match what vLLM's chat
    template produces, but tag attribution and the Controller.Lookup
    inputs only need consistent tokenization on the proxy side."""
    tokens: list[int] = []
    tags: list[str] = []

    sys_b = body.get("system")
    if isinstance(sys_b, str):
        _emit_with_tag(sys_b, "system_static", tokenizer_name, tokens, tags)
    elif isinstance(sys_b, list):
        for b in sys_b:
            if b.get("type") == "text":
                text = b.get("text", "")
                if SKILL_FRONTMATTER_RE.match(text):
                    _emit_text_block_split_dynamic(text, "skill_body", tokenizer_name, tokens, tags)
                else:
                    _emit_text_block_split_dynamic(text, "system_static", tokenizer_name, tokens, tags)

    for tool in body.get("tools") or []:
        encoded = json.dumps(tool, sort_keys=True)
        _emit_with_tag(encoded, "tool_def", tokenizer_name, tokens, tags)

    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            _emit_text_block_split_dynamic(c, "other", tokenizer_name, tokens, tags)
        elif isinstance(c, list):
            for blk in c:
                t = blk.get("type")
                if t == "text":
                    text = blk.get("text", "")
                    if SKILL_FRONTMATTER_RE.match(text):
                        _emit_text_block_split_dynamic(text, "skill_body", tokenizer_name, tokens, tags)
                    else:
                        _emit_text_block_split_dynamic(text, "other", tokenizer_name, tokens, tags)
                elif t == "tool_use":
                    _emit_with_tag(json.dumps(blk, sort_keys=True), "other", tokenizer_name, tokens, tags)
                elif t == "tool_result":
                    content = blk.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
                    _emit_with_tag(str(content), "dynamic", tokenizer_name, tokens, tags)

    return tokens, tags


def assemble_and_tokenize_with_index(
    body: dict, tokenizer_name: str, index: "SkillPrefixIndex | None",
) -> tuple[list[int], list[str]]:
    """Same as assemble_and_tokenize but consults a SkillPrefixIndex for
    embedded skill detection. Falls back to assemble_and_tokenize if
    index is None.

    Algorithm: for each text block, find matching skill prefixes via the
    index. Split the block at match boundaries; tag each segment as
    skill_body inside a match, else use the block's natural base tag."""
    if index is None:
        return assemble_and_tokenize(body, tokenizer_name)

    tokens: list[int] = []
    tags: list[str] = []

    sys_b = body.get("system")
    if isinstance(sys_b, str):
        _emit_block_with_index(sys_b, "system_static", index, tokenizer_name, tokens, tags)
    elif isinstance(sys_b, list):
        for b in sys_b:
            if b.get("type") == "text":
                text = b.get("text", "")
                base = "skill_body" if SKILL_FRONTMATTER_RE.match(text) else "system_static"
                _emit_block_with_index(text, base, index, tokenizer_name, tokens, tags)

    for tool in body.get("tools") or []:
        encoded = json.dumps(tool, sort_keys=True)
        _emit_with_tag(encoded, "tool_def", tokenizer_name, tokens, tags)

    for m in body.get("messages") or []:
        c = m.get("content")
        if isinstance(c, str):
            _emit_block_with_index(c, "other", index, tokenizer_name, tokens, tags)
        elif isinstance(c, list):
            for blk in c:
                t = blk.get("type")
                if t == "text":
                    text = blk.get("text", "")
                    base = "skill_body" if SKILL_FRONTMATTER_RE.match(text) else "other"
                    _emit_block_with_index(text, base, index, tokenizer_name, tokens, tags)
                elif t == "tool_use":
                    _emit_with_tag(json.dumps(blk, sort_keys=True), "other", tokenizer_name, tokens, tags)
                elif t == "tool_result":
                    content = blk.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(p.get("text", "") for p in content if p.get("type") == "text")
                    _emit_with_tag(str(content), "dynamic", tokenizer_name, tokens, tags)

    return tokens, tags


def _emit_block_with_index(text: str, base_tag: str, index, tokenizer_name: str,
                            out_tokens: list[int], out_tags: list[str]) -> None:
    """Find embedded skill prefixes via the index; split the text on match
    boundaries; tag matched ranges as skill_body, surrounding text as
    base_tag (with !`cmd` markers separated as dynamic)."""
    raw = text.encode("utf-8")
    matches = sorted(index.find_matches(raw), key=lambda m: m.byte_start)
    cursor = 0
    for match in matches:
        if match.byte_start > cursor:
            pre = raw[cursor:match.byte_start].decode("utf-8", errors="replace")
            _emit_text_block_split_dynamic(pre, base_tag, tokenizer_name, out_tokens, out_tags)
        skill_chunk = raw[match.byte_start:match.byte_end].decode("utf-8", errors="replace")
        _emit_with_tag(skill_chunk, "skill_body", tokenizer_name, out_tokens, out_tags)
        cursor = match.byte_end
    if cursor < len(raw):
        rest = raw[cursor:].decode("utf-8", errors="replace")
        _emit_text_block_split_dynamic(rest, base_tag, tokenizer_name, out_tokens, out_tags)


def byte_range_to_token_range(
    text: str, byte_start: int, byte_end: int, tokenizer_name: str,
) -> tuple[int, int]:
    """Convert a byte range within `text` to a (token_start, token_end)
    range using the tokenizer's offset_mapping.

    Empty ranges (byte_start == byte_end) return (0, 0). Ranges that fall
    inside a single token's byte span snap to that token boundary."""
    if byte_start == byte_end:
        return (0, 0)
    tok = _tokenizer(tokenizer_name)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets: list[tuple[int, int]] = enc["offset_mapping"]
    tok_start = None
    tok_end = None
    for i, (b0, b1) in enumerate(offsets):
        if tok_start is None and b1 > byte_start:
            tok_start = i
        if b0 < byte_end:
            tok_end = i + 1
    if tok_start is None:
        tok_start = 0
    if tok_end is None:
        tok_end = 0
    return (tok_start, tok_end)
