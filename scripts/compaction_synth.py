"""the harness synthesis fallback — generate a post-compaction request shape
when real `/compact` against Llama-70B errors out or produces unparseable
summaries.

The  spike showed `/compact` works in `--print --continue` mode
on a hosted Anthropic-API model, but Llama-70B may not produce CC's
expected response format. This script is the fallback path.

Pipeline:
1. Read a source capture (a `traces.sqlite` directory plus per-request
   parquet/JSON request bodies). Reconstruct the multi-turn conversation.
2. Call a backend model (any litellm-compatible endpoint) with a CC-style
   "summarize this conversation" system prompt. Capture the summary text.
3. Emit a synthesized request body in CC's post-compact shape:
     - Original system prompt (verbatim)
     - "Conversation summary: \\n{summary_text}\\n" appended
     - One continuation user message
4. Write to a fixture dir alongside real `/compact` captures with
   `meta.json` declaring `compaction_source=synthetic_post_compact`.

The bench harness replay path is identical regardless of source — only the
provenance tag differs.
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger("compaction_synth")

# CC's autocompact system prompt (verbatim from the binary inspection on
# ). Used to drive the summary model so the output shape matches
# what natural autocompact produces.
SUMMARIZER_SYSTEM_PROMPT = (
    "Auto-compact summarizes the conversation when context usage approaches "
    "the model's window. Produce a concise multi-paragraph summary of the "
    "preceding conversation. Preserve: (a) user goals and intent, (b) any "
    "decisions or commitments, (c) file paths and code identifiers mentioned, "
    "(d) outstanding tasks. Drop low-signal scaffolding (greetings, "
    "transitional phrases, repeated explanations). Output the summary text "
    "directly, no preamble."
)

# Format that downstream consumers (bench harness, /compact-shape-detector)
# look for. Mirrors what CC's natural autocompact prepends to the next-turn
# system message.
SUMMARY_BLOCK_TEMPLATE = "\n\nConversation summary: \n{summary_text}\n"


@dataclass
class ReconstructedTurn:
    role: str  # "user" | "assistant"
    content: str
    request_body: dict[str, Any] | None = None


def reconstruct_conversation(source_dir: Path) -> list[ReconstructedTurn]:
    """Read a capture's traces.sqlite + request bodies and reconstruct the
    chronological turn-by-turn conversation. The proxy logs the *full*
    request body the client sent on each call, so we can pull
    `messages[*]` from the latest request and use that as the conversation
    state at the moment of capture."""
    db_path = source_dir / "traces.sqlite"
    if not db_path.exists():
        raise FileNotFoundError(f"no traces.sqlite at {db_path}")

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = list(conn.execute(
            "SELECT request_body_json, ts_start FROM requests ORDER BY ts_start"
        ))
    if not rows:
        raise ValueError(f"traces.sqlite at {db_path} has zero requests")

    # The latest request's `messages` array is the most complete view of
    # the conversation up to that point — every prior turn has been folded
    # into it by claude --resume.
    latest = json.loads(rows[-1]["request_body_json"])
    messages = latest.get("messages", [])

    turns: list[ReconstructedTurn] = []
    for m in messages:
        role = m.get("role", "")
        content = m.get("content", "")
        if isinstance(content, list):
            # Anthropic format: list of {type, text} blocks. Concatenate text.
            content = "".join(
                b.get("text", "") for b in content if b.get("type") == "text"
            )
        turns.append(ReconstructedTurn(role=role, content=content))
    return turns


def extract_system_prompt(source_dir: Path) -> str:
    """Pull the most-recent request's `system` field. CC's `system` carries
    the agent's tool definitions + base instructions; the post-compact
    request must preserve this verbatim."""
    db_path = source_dir / "traces.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT request_body_json FROM requests ORDER BY ts_start DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return ""
    body = json.loads(row["request_body_json"])
    system = body.get("system", "")
    if isinstance(system, list):
        # Anthropic format: list of {type, text} blocks.
        system = "".join(
            b.get("text", "") for b in system if b.get("type") == "text"
        )
    return system or ""


def _default_summarize(
    messages: list[ReconstructedTurn],
    *,
    model: str,
    api_base: str,
    api_key: str,
) -> str:
    """Call litellm to generate the summary. Imported lazily so unit tests
    can mock the whole thing without paying the litellm import cost."""
    import litellm
    chat_messages = [{"role": "system", "content": SUMMARIZER_SYSTEM_PROMPT}]
    for t in messages:
        chat_messages.append({"role": t.role, "content": t.content})
    chat_messages.append({
        "role": "user",
        "content": "Now produce the autocompact summary of the conversation above.",
    })
    resp = litellm.completion(
        model=model,
        messages=chat_messages,
        api_base=api_base,
        api_key=api_key,
        max_tokens=1024,
        temperature=0.0,
    )
    return resp["choices"][0]["message"]["content"].strip()


def build_postcompact_request(
    *,
    original_system: str,
    summary_text: str,
    continuation_user_text: str,
    model: str = "claude-sonnet-4",
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Build the synthesized post-compaction request body. Shape mirrors
    what natural autocompact emits: the original system prompt with the
    `Conversation summary:` block appended, plus a single continuation
    user message."""
    augmented_system = original_system + SUMMARY_BLOCK_TEMPLATE.format(
        summary_text=summary_text
    )
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": augmented_system,
        "messages": [
            {"role": "user", "content": continuation_user_text},
        ],
    }


def synthesize(
    source_dir: Path,
    dest_dir: Path,
    *,
    summarize_fn: Callable[[list[ReconstructedTurn]], str] | None = None,
    summary_model: str = "openai/gpt-4o-mini",
    summary_api_base: str | None = None,
    summary_api_key: str | None = None,
    continuation_text: str = "Now continue from where we left off — what's the next concrete step?",
    target_model: str = "claude-sonnet-4",
) -> Path:
    """Read source capture, generate summary, emit synthesized post-compact
    request body + meta.json under dest_dir. Returns dest_dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    turns = reconstruct_conversation(source_dir)
    original_system = extract_system_prompt(source_dir)

    if summarize_fn is None:
        if summary_api_base is None or summary_api_key is None:
            raise ValueError(
                "summarize_fn unset — provide summary_api_base + summary_api_key "
                "for the default litellm-based summarizer, or pass a custom "
                "summarize_fn (e.g., for tests)"
            )
        def _bound(msgs):
            return _default_summarize(
                msgs, model=summary_model, api_base=summary_api_base,
                api_key=summary_api_key,
            )
        summarize_fn = _bound

    summary_text = summarize_fn(turns)
    log.info("summary length: %d chars", len(summary_text))

    post_compact_body = build_postcompact_request(
        original_system=original_system,
        summary_text=summary_text,
        continuation_user_text=continuation_text,
        model=target_model,
    )

    request_id = f"synth_{uuid.uuid4().hex[:12]}"
    (dest_dir / "post_compact_request.json").write_text(
        json.dumps(post_compact_body, indent=2)
    )
    (dest_dir / "summary.txt").write_text(summary_text)
    (dest_dir / "meta.json").write_text(json.dumps({
        "compaction_source": "synthetic_post_compact",
        "source_capture": str(source_dir),
        "summary_model": summary_model,
        "target_model": target_model,
        "request_id": request_id,
        "synthesized_at": time.time(),
        "schema_version": "plan4_postcompact_v1",
    }, indent=2))
    return dest_dir


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--source", required=True, type=Path,
        help="path to a capture dir containing traces.sqlite",
    )
    p.add_argument(
        "--dest", required=True, type=Path,
        help="output dir for the synthesized post-compact request shape",
    )
    p.add_argument(
        "--summary-model", default="openai/gpt-4o-mini",
        help="litellm model identifier for the summary generation step",
    )
    p.add_argument(
        "--summary-api-base", default=None,
        help="API base URL for the summary model (e.g., http://127.0.0.1:4000)",
    )
    p.add_argument(
        "--summary-api-key", default=None,
        help="API key for the summary model (env: SUMMARY_API_KEY)",
    )
    p.add_argument(
        "--continuation",
        default="Now continue from where we left off — what's the next concrete step?",
        help="text of the post-compact continuation user message",
    )
    p.add_argument(
        "--target-model", default="claude-sonnet-4",
        help="`model` field on the synthesized request body (passed through to bench replay)",
    )
    args = p.parse_args(argv[1:])
    logging.basicConfig(level=logging.INFO)

    import os
    api_key = args.summary_api_key or os.environ.get("SUMMARY_API_KEY")
    if api_key is None or args.summary_api_base is None:
        log.error("--summary-api-base and --summary-api-key (or SUMMARY_API_KEY env) are required")
        return 2

    dest = synthesize(
        args.source, args.dest,
        summary_model=args.summary_model,
        summary_api_base=args.summary_api_base,
        summary_api_key=api_key,
        continuation_text=args.continuation,
        target_model=args.target_model,
    )
    print(f"wrote synthesized post-compact fixture to {dest}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
