"""Unit tests for the extended_30 fixture builder (the harness Layer 2)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from skillcacher.bench.permuted_fixture import (
    BLEND_SEP, Passage, load_library, load_questions,
)
from skillcacher.bench.permuted_extended import (
    N_PASSAGES, N_REQUESTS, generate_extended_30, write_extended_30_fixture,
)


MOCK_JSONL = Path(__file__).parent.parent / "fixtures/mtrag_mock/reference_RAG.jsonl"


def _passages(n: int = 5) -> list[Passage]:
    return [Passage(title=f"P{i}", text=f"body of passage {i} " * 30) for i in range(n)]


def _questions(n: int) -> list[str]:
    return [f"What about passage {i}?" for i in range(n)]


def _segment_hashes(body: dict) -> list[str]:
    """Hash each BLEND_SEP-delimited segment of the user content. Drops the
    leading empty split that comes from content starting with a separator."""
    parts = body["messages"][0]["content"].split(BLEND_SEP)
    return [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in parts if p]


def test_generate_extended_30_produces_30_bodies():
    bodies = generate_extended_30(_passages(), _questions(30), seed=0)
    assert len(bodies) == N_REQUESTS


def test_extended_30_uses_same_5_passages_in_every_request():
    """Every request must contain exactly the same 5 passages — only the
    order varies. The manifest's library is library[:5]; the runtime
    invariant is that the set of passage-content hashes per request is
    identical across all 30 requests."""
    bodies = generate_extended_30(_passages(), _questions(30), seed=1)
    per_request_passage_hashes = []
    for body in bodies:
        # Drop the leading "Reference passages:" wrapper segment + trailing
        # "Question:" wrapper segment; what's left are the N_PASSAGES passage
        # segments in some order.
        segs = _segment_hashes(body)
        # First seg = "Reference passages:"; last seg = "Question: ...".
        passage_segs = set(segs[1:-1])
        assert len(passage_segs) == N_PASSAGES
        per_request_passage_hashes.append(passage_segs)
    # Every request has the identical 5-element set of passage hashes.
    common = set.intersection(*per_request_passage_hashes)
    assert len(common) == N_PASSAGES, \
        f"passages don't match across requests: intersection has {len(common)}"


def test_extended_30_orderings_are_all_distinct():
    """5 passages yield 120 possible orderings; 30 requests should be 30
    distinct orderings (the generator dedups internally)."""
    bodies = generate_extended_30(_passages(), _questions(30), seed=2)
    orderings = []
    for body in bodies:
        segs = _segment_hashes(body)
        orderings.append(tuple(segs[1:-1]))
    assert len(set(orderings)) == N_REQUESTS


def test_extended_30_is_deterministic_under_seed():
    a = generate_extended_30(_passages(), _questions(30), seed=42)
    b = generate_extended_30(_passages(), _questions(30), seed=42)
    a_orders = [_segment_hashes(x)[1:-1] for x in a]
    b_orders = [_segment_hashes(x)[1:-1] for x in b]
    assert a_orders == b_orders


def test_extended_30_cycles_questions_when_fewer_than_30():
    """Cloud corpus may have fewer than 30 distinct questions; the builder
    cycles through what's available (cacheblend behavior is invariant to
    question text — only passages need to repeat)."""
    five_qs = _questions(5)
    bodies = generate_extended_30(_passages(), five_qs, seed=3)
    used_questions = set()
    for body in bodies:
        # Question text is everything after the final " Question: ".
        content = body["messages"][0]["content"]
        q_idx = content.rfind("Question: ")
        used_questions.add(content[q_idx + len("Question: "):])
    # Only the 5 input questions should appear.
    assert used_questions == set(five_qs)


def test_extended_30_raises_on_too_few_passages():
    with pytest.raises(ValueError, match="library passages"):
        generate_extended_30(_passages(n=3), _questions(30), seed=0)


def test_extended_30_raises_on_no_questions():
    with pytest.raises(ValueError, match="at least 1 question"):
        generate_extended_30(_passages(), [], seed=0)


def test_write_extended_30_fixture_layout(tmp_path: Path):
    """End-to-end: real MTRAG mock library → 30 raw_requests + manifest."""
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)  # only 20 distinct in mock; cycle
    write_extended_30_fixture(tmp_path, library=lib, questions=qs, seed=5)

    raw_files = sorted((tmp_path / "raw_requests").glob("*.json"))
    assert len(raw_files) == N_REQUESTS
    assert raw_files[0].name == "000000.json"
    assert raw_files[-1].name == "000029.json"

    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["mode"] == "extended_30"
    assert manifest["seed"] == 5
    assert len(manifest["library"]) == N_PASSAGES
    assert len(manifest["requests"]) == N_REQUESTS
    # Each request entry has the per-segment hashes for reviewer re-derivation.
    assert all("expected_segment_hashes" in r for r in manifest["requests"])


def test_write_extended_30_wipes_stale(tmp_path: Path):
    """Re-running the writer should not leave orphan files from prior runs."""
    raw_dir = tmp_path / "raw_requests"
    raw_dir.mkdir()
    (raw_dir / "999999.json").write_text("{}")

    write_extended_30_fixture(
        tmp_path, library=_passages(), questions=_questions(30), seed=0,
    )
    surviving = sorted(p.name for p in raw_dir.glob("*.json"))
    assert "999999.json" not in surviving
    assert len(surviving) == N_REQUESTS
