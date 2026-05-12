"""span_tagger v2: known-skill-prefix detection inside arbitrary text blocks."""
from pathlib import Path

import pytest

from skillcacher.tagging.skill_prefix_index import SkillPrefixIndex
from skillcacher.proxy.assemble import assemble_and_tokenize_with_index


TEST_TOKENIZER = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    s = d / "useful_skill"
    s.mkdir()
    (s / "SKILL.md").write_text(
        "---\nname: useful_skill\n---\n\n# Useful Skill\n\n"
        + ("Step-by-step guidance. " * 100)
    )
    return d


def test_embedded_skill_prefix_tagged(skills_dir):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024])
    entry = idx.entries()[0]
    # Construct a system block that wraps the skill prefix in summary text
    # (simulating post-compaction CC system block shape).
    summary = "Compaction summary: the user has been refactoring foo.py. " * 10
    after = "\n\nNext step suggestions: ..."
    embedded = summary.encode() + entry.rendered_prefix + after.encode()
    body = {
        "model": "x",
        "system": [{"type": "text", "text": embedded.decode("utf-8", errors="replace")}],
        "messages": [{"role": "user", "content": "continue"}],
        "max_tokens": 1,
    }
    tokens, tags = assemble_and_tokenize_with_index(body, TEST_TOKENIZER, idx)
    assert "skill_body" in tags, "embedded skill prefix should be tagged"
    # The summary preceding the skill should still be system_static (or other)
    assert any(t == "system_static" for t in tags)


def test_no_index_falls_back_to_v1_behavior(skills_dir):
    # No prefix matches in the haystack → no skill_body tags from index
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024])
    body = {
        "model": "x",
        "system": "plain old system prompt with no skills embedded",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
    }
    tokens, tags = assemble_and_tokenize_with_index(body, TEST_TOKENIZER, idx)
    # Either nothing is tagged skill_body, or the fallback frontmatter detector
    # didn't match (correctly).
    assert "skill_body" not in tags
