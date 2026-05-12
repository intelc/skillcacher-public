"""Unit tests for permuted_fixture loaders + generators."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from skillcacher.bench.permuted_fixture import (
    BLEND_SEP, build_body, generate_full_5, generate_partial_20,
    load_library, load_questions, Passage, write_fixture_set,
)

MOCK_JSONL = Path(__file__).parent.parent / "fixtures/mtrag_mock/reference_RAG.jsonl"


def test_load_library_returns_10_passages_from_2_records():
    lib = load_library(MOCK_JSONL)
    assert len(lib) == 10
    # First 5 from R0, next 5 from R1, in record order.
    assert lib[0].title == "R0-Passage0"
    assert lib[4].title == "R0-Passage4"
    assert lib[5].title == "R1-Passage0"
    assert lib[9].title == "R1-Passage4"


def test_load_library_skips_non_cloud_and_turn_2():
    lib = load_library(MOCK_JSONL)
    # The "different-collection" + "turn 2" records contribute zero passages.
    titles = {p.title for p in lib}
    assert all(t.startswith("R") for t in titles)


def test_load_library_passage_text_starts_with_title_marker():
    """The on-the-wire passage format is `[Title]\\n<text>` so each passage
    is a self-describing segment when joined with BLEND_SEP."""
    lib = load_library(MOCK_JSONL)
    assert lib[0].formatted.startswith("[R0-Passage0]\n")


def test_load_questions_returns_20_distinct():
    qs = load_questions(MOCK_JSONL, n=20)
    assert len(qs) == 20
    assert len(set(qs)) == 20  # all distinct


def test_load_questions_raises_when_too_few():
    with pytest.raises(RuntimeError, match="distinct questions"):
        load_questions(MOCK_JSONL, n=99)


def test_load_library_skips_records_with_fewer_than_5_contexts():
    """The mock JSONL contains a 'sparse' cloud turn-1 record with only 3
    contexts. load_library must skip it and still produce 10 passages from
    the two records that DO have >=5 contexts (R0 and R1)."""
    lib = load_library(MOCK_JSONL)
    titles = [p.title for p in lib]
    # No Sparse-Passage* should appear.
    assert not any(t.startswith("Sparse-") for t in titles), \
        f"sparse passages leaked into library: {titles}"
    # Library is still 10, drawn from R0 and R1.
    assert len(lib) == 10
    assert {t.split("-")[0] for t in titles} == {"R0", "R1"}


def _three_passages():
    return [
        Passage(title="A", text="alpha text " * 50),
        Passage(title="B", text="beta text " * 50),
        Passage(title="C", text="gamma text " * 50),
    ]


def test_build_body_shape_is_anthropic_chat_completions():
    body = build_body(_three_passages(), "What is alpha?")
    assert body["model"] == "Qwen/Qwen3-8B"
    assert body["max_tokens"] == 64
    assert isinstance(body["system"], list)
    assert body["system"][0]["type"] == "text"
    assert len(body["messages"]) == 1
    assert body["messages"][0]["role"] == "user"


def test_build_body_passages_first_question_last():
    """Passages must precede the question; trailing question is the only
    segment that varies between requests in a real run."""
    body = build_body(_three_passages(), "What is alpha?")
    content = body["messages"][0]["content"]
    pq = content.find("Question:")
    assert pq > 0  # Question marker is present
    # Each formatted passage's title must appear BEFORE the Question marker.
    for title in ("[A]", "[B]", "[C]"):
        assert content.find(title) < pq, f"{title} did not precede Question:"


def test_build_body_separator_between_every_segment():
    """BLEND_SEP appears: (a) before the first passage, (b) between
    consecutive passages, (c) between the passage block and Question:."""
    body = build_body(_three_passages(), "Q")
    content = body["messages"][0]["content"]
    # 3 passages → 3 leading-or-between separators + 1 separator before Question
    # = 4 occurrences of BLEND_SEP minimum.
    assert content.count(BLEND_SEP) >= 4


def _segment_hashes(body: dict) -> list[str]:
    """Hash each BLEND_SEP-delimited segment of the user content."""
    import hashlib
    content = body["messages"][0]["content"]
    parts = content.split(BLEND_SEP)
    # Drop empty leading split (content starts with "Reference passages:" + sep).
    parts = [p for p in parts if p]
    return [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in parts]


def test_full_5_produces_5_bodies():
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=5)
    bodies = generate_full_5(lib, qs, seed=42)
    assert len(bodies) == 5


def test_full_5_all_passages_appear_every_request():
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=5)
    bodies = generate_full_5(lib, qs, seed=42)
    # Drop the first ("Reference passages:") and last ("Question: ...") segments
    # — keep only the passage segments. With 5 passages, that's 5 hashes.
    passage_hash_sets = []
    for b in bodies:
        all_hashes = _segment_hashes(b)
        # Slice [1:-1] removes "Reference passages:" prefix-segment and the
        # trailing "Question: ..." segment, leaving the 5 passage-segments.
        passage_hash_sets.append(set(all_hashes[1:-1]))
    # All 5 requests must share the SAME 5 passage hashes.
    for s in passage_hash_sets[1:]:
        assert s == passage_hash_sets[0], "full_5 should reuse the same 5 passages"


def test_full_5_orders_differ_across_requests():
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=5)
    bodies = generate_full_5(lib, qs, seed=42)
    orders = [tuple(_segment_hashes(b)[1:-1]) for b in bodies]
    # No two requests share the same passage permutation.
    assert len(set(orders)) == len(orders), f"some orders repeated: {orders}"


def test_full_5_seed_determinism():
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=5)
    a = generate_full_5(lib, qs, seed=42)
    b = generate_full_5(lib, qs, seed=42)
    assert a == b
    c = generate_full_5(lib, qs, seed=43)
    assert c != a


def test_partial_20_produces_20_bodies():
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)
    bodies = generate_partial_20(lib, qs, seed=42)
    assert len(bodies) == 20


def test_partial_20_each_request_has_5_passages():
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)
    bodies = generate_partial_20(lib, qs, seed=42)
    for b in bodies:
        # 1 prefix segment + 5 passage segments + 1 question segment = 7 hashes.
        assert len(_segment_hashes(b)) == 7, f"expected 7 segments, got {_segment_hashes(b)}"


def test_partial_20_intersection_at_least_4():
    """Intersection of passage-segment hashes across all 20 requests >= 4
    (the 'core' set always shared)."""
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)
    bodies = generate_partial_20(lib, qs, seed=42)
    passage_sets = [set(_segment_hashes(b)[1:-1]) for b in bodies]
    intersection = set.intersection(*passage_sets)
    assert len(intersection) >= 4, f"intersection size {len(intersection)} < 4"


def test_partial_20_union_at_most_10():
    """Union of passage-segment hashes <= 10 (the library size)."""
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)
    bodies = generate_partial_20(lib, qs, seed=42)
    passage_sets = [set(_segment_hashes(b)[1:-1]) for b in bodies]
    union = set.union(*passage_sets)
    assert len(union) <= 10, f"union size {len(union)} > 10"


def test_partial_20_questions_distinct():
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)
    bodies = generate_partial_20(lib, qs, seed=42)
    questions_seen = []
    for b in bodies:
        content = b["messages"][0]["content"]
        q_idx = content.rfind("Question: ")
        questions_seen.append(content[q_idx:])
    assert len(set(questions_seen)) == 20


def test_partial_20_seed_determinism():
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)
    a = generate_partial_20(lib, qs, seed=42)
    b = generate_partial_20(lib, qs, seed=42)
    assert a == b
    c = generate_partial_20(lib, qs, seed=43)
    assert c != a


def test_write_fixture_set_full_5(tmp_path):
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=5)
    write_fixture_set(tmp_path, mode="full_5", library=lib, questions=qs, seed=42)
    raw = tmp_path / "raw_requests"
    files = sorted(raw.glob("*.json"))
    assert len(files) == 5
    assert files[0].name == "000000.json"
    assert files[-1].name == "000004.json"
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["mode"] == "full_5"
    assert manifest["seed"] == 42
    assert len(manifest["library"]) == 10
    assert len(manifest["requests"]) == 5


def test_write_fixture_set_partial_20(tmp_path):
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)
    write_fixture_set(tmp_path, mode="partial_20", library=lib, questions=qs, seed=42)
    files = sorted((tmp_path / "raw_requests").glob("*.json"))
    assert len(files) == 20


def test_manifest_hashes_match_body_segments(tmp_path):
    """For each request, manifest.requests[i].expected_segment_hashes must
    equal the actual sha256 of each BLEND_SEP-split segment in the body."""
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=20)
    write_fixture_set(tmp_path, mode="partial_20", library=lib, questions=qs, seed=42)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    for entry in manifest["requests"]:
        idx = entry["idx"]
        body = json.loads((tmp_path / "raw_requests" / f"{idx:06d}.json").read_text())
        actual = _segment_hashes(body)
        assert actual == entry["expected_segment_hashes"]


def test_write_fixture_set_idempotent_under_same_seed(tmp_path):
    """Re-running with the same seed produces byte-identical output."""
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=5)
    a_dir = tmp_path / "a"
    b_dir = tmp_path / "b"
    write_fixture_set(a_dir, mode="full_5", library=lib, questions=qs, seed=42)
    write_fixture_set(b_dir, mode="full_5", library=lib, questions=qs, seed=42)
    for i in range(5):
        a = (a_dir / "raw_requests" / f"{i:06d}.json").read_bytes()
        b = (b_dir / "raw_requests" / f"{i:06d}.json").read_bytes()
        assert a == b
    # Manifests should also be byte-identical.
    assert (a_dir / "manifest.json").read_bytes() == (b_dir / "manifest.json").read_bytes()


def test_write_fixture_set_wipes_stale_files(tmp_path):
    """Stale raw_requests/*.json from a prior run should be removed."""
    raw = tmp_path / "raw_requests"
    raw.mkdir(parents=True)
    (raw / "999999.json").write_text("{}")
    lib = load_library(MOCK_JSONL)
    qs = load_questions(MOCK_JSONL, n=5)
    write_fixture_set(tmp_path, mode="full_5", library=lib, questions=qs, seed=42)
    assert not (raw / "999999.json").exists()
    assert (raw / "000000.json").exists()
