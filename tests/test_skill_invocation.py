"""offline fixture roundtrip for the skill_invocation builder.

Loads `tests/fixtures/test_skills/`, asks `scripts.skill_invocation_prompts`
to build prompts, and asserts that:

1. exactly 5 skills × 3 prompts = 15 prompts are produced;
2. each prompt mentions its skill_id by name;
3. each prompt contains a verbatim ≥1024-byte slice of its target skill's
   rendered body;
4. each prompt triggers a ≥1024-byte own-skill match in `SkillPrefixIndex`
   (this is the actual signal the proxy uses for cacheblend retrieval).

Pure-offline; runs in milliseconds and acts as a pre-flight gate before
any pod spend on the §2 captures.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.skill_invocation_prompts import build_prompts
from skillcacher.tagging.skill_prefix_index import SkillPrefixIndex, render_skill_body


SKILL_DIR = Path(__file__).parent / "fixtures" / "test_skills"
EXPECTED_SKILLS = {
    "weather-format",
    "commit-msg-style",
    "bash-safety",
    "python-import-order",
    "markdown-table",
}
PROMPTS_PER_SKILL = 3
SMALLEST_ANCHOR = 1024


@pytest.fixture(scope="module")
def prompts():
    return build_prompts([SKILL_DIR])


@pytest.fixture(scope="module")
def prefix_index() -> SkillPrefixIndex:
    return SkillPrefixIndex([SKILL_DIR])


def test_skill_fixtures_exist():
    found = {p.parent.name for p in SKILL_DIR.glob("*/SKILL.md")}
    assert found == EXPECTED_SKILLS, (
        f"expected exactly {EXPECTED_SKILLS} under {SKILL_DIR}, found {found}"
    )


def test_each_skill_renders_above_smallest_anchor():
    """Each rendered body must be ≥ smallest anchor (1024) so the prefix
    index actually indexes it. If a body falls under the floor, the demo
    is silently broken."""
    for skill_md in SKILL_DIR.glob("*/SKILL.md"):
        rendered = render_skill_body(skill_md)
        assert len(rendered) >= SMALLEST_ANCHOR, (
            f"{skill_md.parent.name} rendered body is {len(rendered)} bytes; "
            f"need ≥ {SMALLEST_ANCHOR}"
        )


def test_prompt_count_and_coverage(prompts):
    assert len(prompts) == len(EXPECTED_SKILLS) * PROMPTS_PER_SKILL
    by_skill: dict[str, list] = {}
    for ip in prompts:
        by_skill.setdefault(ip.skill_id, []).append(ip)
    assert set(by_skill) == EXPECTED_SKILLS
    for skill, items in by_skill.items():
        assert len(items) == PROMPTS_PER_SKILL, f"{skill}: {len(items)} prompts"
        prompt_ids = {ip.prompt_id for ip in items}
        assert len(prompt_ids) == PROMPTS_PER_SKILL, f"{skill}: duplicate prompt_id"


def test_each_prompt_mentions_its_skill_id(prompts):
    for ip in prompts:
        assert ip.skill_id in ip.prompt, (
            f"prompt {ip.task_id} does not mention skill_id {ip.skill_id}"
        )


def test_each_prompt_contains_verbatim_anchor(prompts):
    """The proxy's prefix index does `haystack.find(rendered_prefix)`, so
    the rendered prefix must appear verbatim as a contiguous substring of
    the prompt bytes. Validate exactly that."""
    for ip in prompts:
        skill_md = SKILL_DIR / ip.skill_id / "SKILL.md"
        rendered = render_skill_body(skill_md)
        anchor = rendered[:SMALLEST_ANCHOR]
        assert anchor in ip.prompt.encode("utf-8"), (
            f"prompt {ip.task_id} does not contain a verbatim "
            f"{SMALLEST_ANCHOR}-byte slice of the rendered body"
        )


def test_each_prompt_triggers_prefix_index_match(prompts, prefix_index):
    """Belt-and-suspenders: run the actual `SkillPrefixIndex.find_matches`
    against each prompt and assert it produces a match keyed to the
    correct skill_id at the smallest-anchor or larger."""
    for ip in prompts:
        matches = prefix_index.find_matches(ip.prompt.encode("utf-8"))
        own = [m for m in matches if m.skill_id == ip.skill_id]
        assert own, f"prompt {ip.task_id}: no own-skill match"
        longest = max(m.anchor_bytes for m in own)
        assert longest >= SMALLEST_ANCHOR, (
            f"prompt {ip.task_id}: longest own-skill anchor is {longest}, "
            f"need ≥ {SMALLEST_ANCHOR}"
        )
