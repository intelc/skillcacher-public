"""Structural parser. Tags each prompt segment by kind so the bench harness
can decompose hit rate by skill / tool_def / system / dynamic / other."""
import json
import re
from functools import lru_cache
from typing import Any, Literal

Kind = Literal["system_static", "tool_def", "skill_body", "dynamic", "other"]

SKILL_FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
DYNAMIC_BLOCK_RE = re.compile(r"!`[^`]+`")  # Claude Code !`cmd` injection markers


@lru_cache(maxsize=4)
def _tokenizer(name: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name)


def _count_tokens(text: str, tokenizer_name: str) -> int:
    if not text:
        return 0
    tok = _tokenizer(tokenizer_name)
    return len(tok.encode(text, add_special_tokens=False))


def tag_prompt(req: dict[str, Any], tokenizer_name: str) -> list[tuple[Kind, int]]:
    """Walk the request and return a list of (kind, token_count) covering the prompt in order."""
    out: list[tuple[Kind, int]] = []

    system = req.get("system")
    if isinstance(system, str):
        out.append(("system_static", _count_tokens(system, tokenizer_name)))
    elif isinstance(system, list):
        for block in system:
            if block.get("type") == "text":
                out.append(("system_static", _count_tokens(block.get("text", ""), tokenizer_name)))

    for tool in req.get("tools", []) or []:
        encoded = json.dumps(tool, sort_keys=True)
        out.append(("tool_def", _count_tokens(encoded, tokenizer_name)))

    for m in req.get("messages", []) or []:
        content = m.get("content")
        if isinstance(content, str):
            out.extend(_tag_text(content, tokenizer_name))
        elif isinstance(content, list):
            for c in content:
                if c.get("type") == "text":
                    out.extend(_tag_text(c.get("text", ""), tokenizer_name))
                elif c.get("type") == "tool_use":
                    out.append(("other", _count_tokens(json.dumps(c, sort_keys=True), tokenizer_name)))
                elif c.get("type") == "tool_result":
                    text = c.get("content", "")
                    if isinstance(text, list):
                        text = "\n".join(b.get("text", "") for b in text if b.get("type") == "text")
                    out.append(("dynamic", _count_tokens(str(text), tokenizer_name)))

    return [t for t in out if t[1] > 0]


def _tag_text(text: str, tokenizer_name: str) -> list[tuple[Kind, int]]:
    """Decompose a free-text block into (skill_body | dynamic | other) segments."""
    if not text:
        return []
    if SKILL_FRONTMATTER_RE.match(text):
        # Whole text is a skill body. Split out !`cmd` dynamic regions.
        return _split_dynamic(text, base_kind="skill_body", tokenizer_name=tokenizer_name)
    return _split_dynamic(text, base_kind="other", tokenizer_name=tokenizer_name)


def _split_dynamic(text: str, base_kind: Kind, tokenizer_name: str) -> list[tuple[Kind, int]]:
    pieces: list[tuple[Kind, int]] = []
    cursor = 0
    for m in DYNAMIC_BLOCK_RE.finditer(text):
        if m.start() > cursor:
            pieces.append((base_kind, _count_tokens(text[cursor:m.start()], tokenizer_name)))
        pieces.append(("dynamic", _count_tokens(m.group(0), tokenizer_name)))
        cursor = m.end()
    if cursor < len(text):
        pieces.append((base_kind, _count_tokens(text[cursor:], tokenizer_name)))
    return pieces
