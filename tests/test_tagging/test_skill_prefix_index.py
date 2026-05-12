"""Tests for the skill prefix index — discovery, rendering, hash matching."""
import pytest
from pathlib import Path

from skillcacher.tagging.skill_prefix_index import (
    SkillPrefixIndex, SkillPrefixEntry, render_skill_body,
)


@pytest.fixture
def skills_dir(tmp_path: Path) -> Path:
    d = tmp_path / "skills"
    d.mkdir()
    skill_a = d / "alpha"
    skill_a.mkdir()
    (skill_a / "SKILL.md").write_text(
        "---\n"
        "name: alpha\n"
        "description: alpha skill\n"
        "---\n"
        "\n"
        "# Alpha\n\n"
        + ("This is a long alpha body. " * 200)
    )
    skill_b = d / "beta"
    skill_b.mkdir()
    (skill_b / "SKILL.md").write_text(
        "---\n"
        "name: beta\n"
        "description: beta skill\n"
        "---\n\n"
        "# Beta\n\n"
        + ("Beta body content. " * 70)
    )
    return d


def test_render_skill_body_strips_frontmatter():
    text = "---\nname: x\n---\n\n# Body\n\nhello !`echo dynamic`"
    body = render_skill_body_str(text)
    assert "name: x" not in body
    assert "echo dynamic" not in body  # !`...` markers stripped
    assert "Body" in body
    assert "hello" in body


def render_skill_body_str(text: str) -> str:
    """Test helper that calls the public function on raw text."""
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
        f.write(text)
        f.flush()
        return render_skill_body(Path(f.name)).decode("utf-8")


def test_index_discovers_skills(skills_dir):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024, 2048, 4096])
    entries = list(idx.entries())
    skill_ids = {e.skill_id for e in entries}
    assert skill_ids == {"alpha", "beta"}


def test_index_anchors_per_skill(skills_dir):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024, 2048, 4096])
    alpha_entries = [e for e in idx.entries() if e.skill_id == "alpha"]
    # Alpha's body is ~5KB, all three anchors should fit
    assert sorted(e.anchor_bytes for e in alpha_entries) == [1024, 2048, 4096]


def test_index_skips_anchors_longer_than_body(skills_dir):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024, 2048, 4096])
    beta_entries = [e for e in idx.entries() if e.skill_id == "beta"]
    # Beta body ~1KB; only 1024-anchor fits
    assert [e.anchor_bytes for e in beta_entries] == [1024]


def test_find_matches_locates_anchor_in_haystack(skills_dir):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024, 2048])
    alpha_entry = next(e for e in idx.entries() if e.skill_id == "alpha" and e.anchor_bytes == 2048)
    haystack = b"PREFIX " + alpha_entry.rendered_prefix + b" SUFFIX"
    matches = idx.find_matches(haystack)
    assert any(m.skill_id == "alpha" for m in matches)
    m = next(m for m in matches if m.skill_id == "alpha")
    assert m.byte_start == len(b"PREFIX ")
    assert m.byte_end == m.byte_start + m.anchor_bytes


def test_find_matches_longest_anchor_wins(skills_dir):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024, 2048, 4096])
    alpha_entry = next(e for e in idx.entries() if e.skill_id == "alpha" and e.anchor_bytes == 4096)
    haystack = alpha_entry.rendered_prefix
    matches = idx.find_matches(haystack)
    alpha_matches = [m for m in matches if m.skill_id == "alpha"]
    assert len(alpha_matches) == 1
    assert alpha_matches[0].anchor_bytes == 4096


def test_find_matches_returns_empty_on_no_match(skills_dir):
    idx = SkillPrefixIndex([skills_dir], anchor_bytes=[1024])
    matches = idx.find_matches(b"completely unrelated text " * 50)
    assert matches == []


def test_discover_follows_symlinks(tmp_path: Path):
    """cursor-skills-symlinked-into-`~/.claude/skills/` were
    invisible to Path.rglob in the harness Tier-3. _discover must follow
    symlinks via os.walk(followlinks=True)."""
    real_skills = tmp_path / "real_skills"
    real_skills.mkdir()
    (real_skills / "linked").mkdir()
    (real_skills / "linked" / "SKILL.md").write_text(
        "---\nname: linked\n---\n\n# Linked\n\n"
        + ("Linked body. " * 200)
    )

    discover_root = tmp_path / "skills"
    discover_root.mkdir()
    (discover_root / "linked-from-cursor").symlink_to(real_skills / "linked")

    idx = SkillPrefixIndex([discover_root], anchor_bytes=[1024])
    skill_ids = {e.skill_id for e in idx.entries()}
    assert "linked-from-cursor" in skill_ids


def test_discover_terminates_on_symlink_cycle(tmp_path: Path):
    """P4-R7 mitigation: a directory cycle (A/sub → ../A) must bail rather
    than loop forever. visited-inode tracking handles this."""
    a = tmp_path / "a"
    a.mkdir()
    (a / "SKILL.md").write_text(
        "---\nname: a\n---\n\n# A\n\n" + ("A body. " * 200)
    )
    # Create a cycle: a/loop → a (its own parent).
    (a / "loop").symlink_to(a)

    idx = SkillPrefixIndex([a], anchor_bytes=[1024])
    # Cycle terminates and the SKILL.md still gets discovered exactly once.
    a_entries = [e for e in idx.entries() if e.skill_id == "a"]
    assert len(a_entries) == 1
