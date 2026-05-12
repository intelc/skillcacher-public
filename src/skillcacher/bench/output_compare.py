"""Output-comparison metrics.

Three metrics:
- token_identity_rate(a, b): longest common prefix of canonicalized text /
  max(len). Headline metric for §1 (T=0).
- sampled_set_jaccard(a_samples, b_samples): |A ∩ B| / |A ∪ B| over
  canonicalized strings, sample sets deduped. Headline metric for §2 (T>0).
- modal_position_agreement(a_samples, b_samples): per-position modal-char
  agreement between two sample sets. Supplementary §2 metric. Char-level
  rather than token-level — token-level via tokenizer roundtrip is a
  followup if char-level signal turns out to be insufficient.

Tool-call canonicalization: tool_use blocks get re-serialized with
sort_keys + compact separators so {"a":1,"b":2} and {"b": 2, "a": 1}
compare equal. Whitespace-only diffs in tool-call args don't count as
divergence per design."""
from __future__ import annotations

import json
from collections import Counter
from typing import Sequence

from skillcacher.bench.output_capture import Generation


def canonicalize(gen: Generation) -> str:
    parts: list[str] = []
    for blk in gen.content_blocks:
        kind = blk.get("type")
        if kind == "text":
            parts.append(blk.get("text", ""))
        elif kind == "tool_use":
            payload = {
                "type": "tool_use",
                "name": blk.get("name", ""),
                "input": blk.get("input", {}),
            }
            parts.append(json.dumps(payload, sort_keys=True, separators=(",", ":")))
        else:
            parts.append(json.dumps(blk, sort_keys=True, separators=(",", ":")))
    return "\n".join(parts)


def _lcp_len(a: str, b: str) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def token_identity_rate(a: Generation, b: Generation) -> float:
    """LCP-based identity over canonicalized text.

    Returns 1.0 when both canonicalizations are empty (vacuously
    identical). Otherwise: len(longest_common_prefix) / max(len(ca), len(cb)).
    A divergence in the very first character produces 0.0."""
    ca = canonicalize(a)
    cb = canonicalize(b)
    if not ca and not cb:
        return 1.0
    denom = max(len(ca), len(cb))
    if denom == 0:
        return 1.0
    return _lcp_len(ca, cb) / denom


def sampled_set_jaccard(
    a_samples: Sequence[Generation],
    b_samples: Sequence[Generation],
) -> float:
    """Jaccard over canonicalized sample sets.

    Both sides empty → 1.0. One side empty, one not → 0.0."""
    a_set = {canonicalize(s) for s in a_samples}
    b_set = {canonicalize(s) for s in b_samples}
    if not a_set and not b_set:
        return 1.0
    union = len(a_set | b_set)
    if union == 0:
        return 1.0
    return len(a_set & b_set) / union


def modal_position_agreement(
    a_samples: Sequence[Generation],
    b_samples: Sequence[Generation],
) -> float:
    """Per-position modal-char agreement between two sample sets.

    At each character position, take the modal char across each side's
    sample strings (canonicalized); report fraction of positions where
    the two modal chars agree. Length is the smaller of the two sides'
    longest sample, so a side that systematically truncates earlier
    won't be penalized for positions it doesn't reach.

    Returns 1.0 when both sides are entirely empty; 0.0 when one is empty
    and the other isn't."""
    a_strs = [canonicalize(s) for s in a_samples]
    b_strs = [canonicalize(s) for s in b_samples]
    if not a_strs and not b_strs:
        return 1.0
    if not a_strs or not b_strs:
        return 0.0
    a_max = max(len(s) for s in a_strs)
    b_max = max(len(s) for s in b_strs)
    n = min(a_max, b_max)
    if n == 0:
        return 1.0 if a_max == b_max else 0.0
    agree = 0
    for i in range(n):
        a_modal = Counter(s[i] for s in a_strs if i < len(s)).most_common(1)
        b_modal = Counter(s[i] for s in b_strs if i < len(s)).most_common(1)
        if a_modal and b_modal and a_modal[0][0] == b_modal[0][0]:
            agree += 1
    return agree / n
