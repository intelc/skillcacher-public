import json
from pathlib import Path
from skillcacher.tagging.span_tagger import tag_prompt

FIXTURE = Path(__file__).parent.parent / "fixtures" / "claude_code_simple.json"

# NousResearch mirror is ungated; use it if the Llama-3.3-70B tokenizer
# is inaccessible due to HF gating.
TOKENIZER = "NousResearch/Meta-Llama-3-8B"


def test_tag_prompt_basic():
    req = json.loads(FIXTURE.read_text())
    tags = tag_prompt(req, tokenizer_name=TOKENIZER)
    # Result is a list of (segment_kind, token_count) tuples covering the whole prompt
    kinds = [k for k, _ in tags]
    assert "system_static" in kinds
    assert "tool_def" in kinds
    assert "other" in kinds  # the user message
    total = sum(c for _, c in tags)
    assert total > 0


def test_tag_prompt_with_skill_body():
    req = {
        "model": "x",
        "system": "You are Claude.",
        "messages": [
            {
                "role": "user",
                "content": [{
                    "type": "text",
                    "text": "---\ndescription: test skill\n---\n\n## Instructions\n\nDo the thing.",
                }],
            },
        ],
        "tools": [],
    }
    tags = tag_prompt(req, tokenizer_name=TOKENIZER)
    kinds = [k for k, _ in tags]
    assert "skill_body" in kinds
