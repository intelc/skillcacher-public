"""Byte-prefix index over installed Claude Code skills.

Discovers SKILL.md files under given roots, renders each body the way CC does
(frontmatter parsed out, !`cmd` injection markers stripped), and indexes
the first N bytes at multiple anchor lengths. find_matches() scans an
arbitrary byte haystack for those anchors; longest-anchor matches win.

design spec §3 variant decision (set by an earlier development spike):
- B-as-designed: anchors=[1024, 2048, 4096], longest-anchor-wins
- B-with-multi-anchor: same shape, accept partial matches
- A-fallback: this module is unused; span_tagger keeps frontmatter-only path
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Iterable

FRONTMATTER_RE = re.compile(r"^---\s*\n.*?\n---\s*\n", re.DOTALL)
DYNAMIC_BLOCK_RE = re.compile(r"!`[^`]+`")


def render_skill_body(skill_md: Path) -> bytes:
    """Render a SKILL.md the way Claude Code does — strip frontmatter and
    !`cmd` injection markers; keep everything else byte-for-byte."""
    text = skill_md.read_text()
    text = FRONTMATTER_RE.sub("", text, count=1)
    text = DYNAMIC_BLOCK_RE.sub("", text)
    return text.encode("utf-8")


@dataclass
class SkillPrefixEntry:
    skill_id: str
    anchor_bytes: int
    rendered_prefix: bytes
    hash_digest: bytes


@dataclass
class Match:
    skill_id: str
    anchor_bytes: int
    byte_start: int
    byte_end: int


class SkillPrefixIndex:
    """Indexes skill prefixes for fast substring detection in arbitrary blocks."""

    def __init__(self, skill_dirs: list[Path], *, anchor_bytes: list[int] = (1024, 2048, 4096)):
        self._anchor_bytes = sorted(anchor_bytes)
        self._entries: list[SkillPrefixEntry] = []
        for d in skill_dirs:
            for skill_md in self._discover(d):
                rendered = render_skill_body(skill_md)
                skill_id = skill_md.parent.name
                for n in self._anchor_bytes:
                    if len(rendered) < n:
                        continue
                    prefix = rendered[:n]
                    self._entries.append(SkillPrefixEntry(
                        skill_id=skill_id,
                        anchor_bytes=n,
                        rendered_prefix=prefix,
                        hash_digest=blake2b(prefix, digest_size=16).digest(),
                    ))

    @staticmethod
    def _discover(root: Path) -> Iterable[Path]:
        """Walk `root` recursively, following symlinks, and yield SKILL.md
        files in deterministic order. `Path.rglob` does not follow symlinks,
        so cursor-skills-symlinked-into-`~/.claude/skills/` was invisible
        in the harness's Tier-3 capture .

        Cycles (e.g., A/B → ../A) are bailed on by tracking visited
        directory inodes via (st_dev, st_ino)."""
        if not root.exists():
            return
        visited: set[tuple[int, int]] = set()
        found: list[Path] = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
            try:
                st = os.stat(dirpath)
                key = (st.st_dev, st.st_ino)
            except OSError:
                # Stat may fail on a broken symlink the walk surfaced; skip.
                dirnames.clear()
                continue
            if key in visited:
                # Symlink cycle — bail on this branch.
                dirnames.clear()
                continue
            visited.add(key)
            # Deterministic walk order — os.walk sees dirnames in
            # filesystem-arbitrary order; sorting in-place propagates the
            # ordering downwards.
            dirnames.sort()
            if "SKILL.md" in filenames:
                found.append(Path(dirpath) / "SKILL.md")
        yield from sorted(found)

    def entries(self) -> list[SkillPrefixEntry]:
        return list(self._entries)

    def find_matches(self, haystack: bytes) -> list[Match]:
        """Return all anchor matches in the haystack, longest anchor per
        (skill_id, byte_start) wins.

        For now's expected scale (~50 skills × 3 anchors = 150 needles,
        haystack typically <500KB), naive `bytes.find` per needle is fast
        enough (sub-millisecond). If profiling shows this as hot, swap
        for Aho-Corasick (pyahocorasick)."""
        # Group by skill_id then iterate anchors longest-first; record first match.
        by_skill: dict[str, list[SkillPrefixEntry]] = {}
        for e in self._entries:
            by_skill.setdefault(e.skill_id, []).append(e)
        for entries in by_skill.values():
            entries.sort(key=lambda e: -e.anchor_bytes)

        matches: list[Match] = []
        for skill_id, entries in by_skill.items():
            for entry in entries:
                idx = haystack.find(entry.rendered_prefix)
                if idx == -1:
                    continue
                matches.append(Match(
                    skill_id=skill_id,
                    anchor_bytes=entry.anchor_bytes,
                    byte_start=idx,
                    byte_end=idx + entry.anchor_bytes,
                ))
                break  # longest-anchor wins per skill
        return matches
