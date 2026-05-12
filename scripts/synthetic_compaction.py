"""the harness Layer 3 — synthetic compaction-trigger.

When natural compaction doesn't fire on a SWE-Bench Verified task (the harness flagged this as a real risk on smaller-model setups), this script
emits a stub user message of approximately N tokens. Pasting that into
the live Claude Code session pushes context past the autocompact threshold;
CC's own machinery then produces a real compaction summary on the next
user turn.

Only the *trigger* is synthetic. The compaction *summary* CC produces is
genuine — same code path that fires on natural compaction, same shape, same
re-attached skill prefixes.

Usage:
    python -m scripts.synthetic_compaction --size 250000 > stub.txt
    # Then paste stub.txt into the live CC session.

    python -m scripts.synthetic_compaction --tokenizer meta-llama/Llama-3.3-70B-Instruct \\
        --size 250000 --out stub.txt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Default target: well above CC's reported ~200K compact threshold. Header
# overhead is small (~200 tokens); the rest is filler.
DEFAULT_SIZE_TOKENS = 250_000
# Llama-family rule of thumb: ~4 chars per token. Slightly under-estimates,
# which is the safe direction (resulting text is slightly long).
CHARS_PER_TOKEN_APPROX = 4

HEADER = (
    "## Compaction trigger (synthetic)\n\n"
    "The following is a deterministic block of filler text whose only purpose "
    "is to push the conversation context past the autocompact threshold. "
    "Please ignore the contents. After processing this message, continue with "
    "the previous conversation as if this trigger had not been sent.\n\n"
)
# Repetitive but real-shaped filler. Each line is ~24 tokens on Llama.
FILLER_LINE = (
    "This is a synthetic compaction-trigger filler line. "
    "It is 100% safe to ignore. "
    "The contents will be summarized away by Claude Code's autocompact. "
)


def build_stub_message(
    size_tokens: int = DEFAULT_SIZE_TOKENS,
    *,
    tokenizer_name: str | None = None,
) -> str:
    """Build a stub message of approximately ``size_tokens`` tokens.

    With ``tokenizer_name`` set, count tokens accurately by encoding with
    the model's tokenizer. Without it, approximate at 4 chars per token —
    Llama family rule of thumb. The approximation under-counts slightly,
    which is the safe direction (resulting text is slightly long, still
    crosses the autocompact threshold)."""
    if tokenizer_name is None:
        target_chars = max(size_tokens * CHARS_PER_TOKEN_APPROX - len(HEADER), 0)
        n_lines = target_chars // len(FILLER_LINE) + 1
        return HEADER + (FILLER_LINE * n_lines)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    text = HEADER
    safety_cap = 200_000  # bail if we somehow loop forever
    while len(tok.encode(text, add_special_tokens=False)) < size_tokens:
        text += FILLER_LINE
        safety_cap -= 1
        if safety_cap <= 0:
            break
    return text


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Emit a synthetic compaction-trigger user message."
    )
    parser.add_argument(
        "--size", type=int, default=DEFAULT_SIZE_TOKENS,
        help=f"target token count (default: {DEFAULT_SIZE_TOKENS:,})",
    )
    parser.add_argument(
        "--tokenizer", type=str, default=None,
        help="HF tokenizer name for accurate counting (default: 4-chars/token approximation)",
    )
    parser.add_argument(
        "--out", type=Path, default=None,
        help="write the stub to this file instead of stdout",
    )
    args = parser.parse_args(argv[1:])

    stub = build_stub_message(args.size, tokenizer_name=args.tokenizer)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(stub)
        print(f"wrote {len(stub):,} chars to {args.out}", file=sys.stderr)
    else:
        sys.stdout.write(stub)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
