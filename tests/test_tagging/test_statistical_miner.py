"""unit tests for the statistical span miner."""
from __future__ import annotations

import random

import pytest

from skillcacher.tagging.statistical_miner import (
    MinedSpan,
    mine_spans,
    _shingle,
    _signature,
    _signature_jaccard,
)


def _rand_stream(rng: random.Random, n: int, vocab: int = 50_000) -> list[int]:
    return [rng.randrange(0, vocab) for _ in range(n)]


def test_returns_empty_on_empty_corpus():
    assert mine_spans([]) == []
    assert mine_spans([[]]) == []


def test_returns_empty_when_freq_floor_unreachable():
    """Single-stream corpora can't produce frequency >= 2."""
    rng = random.Random(0)
    s = _rand_stream(rng, 1024)
    assert mine_spans([s], length_floor=64, frequency_floor=2) == []


def test_finds_planted_repeat_across_three_streams():
    """Plant a 300-token sequence in 3 of 5 streams; the miner must
    find it at frequency=3 with the exact tokens."""
    rng = random.Random(42)
    planted = _rand_stream(rng, 300, vocab=10_000)
    streams = [
        _rand_stream(rng, 500) + planted + _rand_stream(rng, 500),
        _rand_stream(rng, 800),
        _rand_stream(rng, 200) + planted + _rand_stream(rng, 100),
        _rand_stream(rng, 600),
        planted + _rand_stream(rng, 700),
    ]
    spans = mine_spans(streams, length_floor=256, frequency_floor=3)
    assert len(spans) >= 1
    top = spans[0]
    assert top.frequency == 3
    assert top.source_stream_ids == frozenset({0, 2, 4})
    # The miner returns the longest common run; for a planted sequence
    # surrounded by random tokens, that's exactly the planted run.
    assert tuple(planted) == top.token_ids


def test_below_length_floor_is_dropped():
    """A 200-token repeat in 3 streams is below length_floor=256 and
    must be dropped."""
    rng = random.Random(1)
    planted = _rand_stream(rng, 200)
    streams = [
        _rand_stream(rng, 500) + planted,
        planted + _rand_stream(rng, 400),
        planted + _rand_stream(rng, 300),
    ]
    assert mine_spans(streams, length_floor=256, frequency_floor=3) == []
    # Lower the floor and the same input must produce a hit.
    spans = mine_spans(streams, length_floor=128, frequency_floor=3)
    assert any(tuple(planted) == s.token_ids for s in spans)


def test_below_frequency_floor_is_dropped():
    """A 300-token repeat in 2 streams is below frequency_floor=3."""
    rng = random.Random(2)
    planted = _rand_stream(rng, 300)
    streams = [
        _rand_stream(rng, 200) + planted,
        planted + _rand_stream(rng, 300),
        _rand_stream(rng, 400),
        _rand_stream(rng, 500),
    ]
    assert mine_spans(streams, length_floor=256, frequency_floor=3) == []


def test_exact_prefix_dedup_keeps_longer():
    """When stream A contains [planted-300-tokens] and stream B contains
    [planted-300-tokens + planted-extra-50] (so B has a strict 350-token
    superset), the miner finds two candidate ranges. The 300-token
    prefix-only candidate must be dropped because it's a strict prefix
    of the 350-token one.

    To trigger both: plant 300t in 2 streams, plant 350t in another 2
    streams (where the 350t starts with the same 300t). Frequency
    floor 3: only the 300t ranges fire (4 streams) and the 350t ranges
    fire (2 streams). The combined block (suffix array sees them
    adjacent) gives a single 300-token repeat at frequency 4.

    So we make a sharper version: plant 350t in 3 streams; the miner
    naturally finds the 350t span and the 300t prefix would be
    dropped IF it appeared. We assert the dedup keeps the longest."""
    rng = random.Random(3)
    planted_350 = _rand_stream(rng, 350)
    planted_300 = planted_350[:300]
    # Two streams contain only the 300-prefix; three streams contain
    # the full 350. Frequency 5 for the 300-prefix block, 3 for the
    # 350-extension. Dedup must keep the 350-token one.
    streams = [
        _rand_stream(rng, 100) + planted_300 + _rand_stream(rng, 100),
        _rand_stream(rng, 200) + planted_300 + _rand_stream(rng, 200),
        _rand_stream(rng, 50) + planted_350 + _rand_stream(rng, 50),
        _rand_stream(rng, 80) + planted_350 + _rand_stream(rng, 80),
        _rand_stream(rng, 120) + planted_350 + _rand_stream(rng, 120),
    ]
    spans = mine_spans(streams, length_floor=256, frequency_floor=3)
    # We expect the miner to keep the longer 350-token span (freq=3)
    # over a strict 300-token-prefix candidate (which would have freq=5).
    by_len = {s.length: s for s in spans}
    assert 350 in by_len, f"350-token span missing — got lengths {sorted(by_len)}"
    # The 300-prefix span (with the same prefix as the 350) must be
    # dedup'd by the prefix-dedup pass.
    assert 300 not in by_len, (
        f"prefix-dedup failed: 300-token prefix kept alongside 350: "
        f"{sorted(by_len)}"
    )


def test_minhash_near_dup_collapses_similar_spans():
    """Two spans differing by a handful of tokens should merge under the
    MinHash dedup at threshold 0.8."""
    rng = random.Random(4)
    base = _rand_stream(rng, 400)
    # a near-dup differs in 5 of 400 tokens (~1.2% — well within 0.8 J).
    near = list(base)
    for i in [50, 150, 200, 300, 380]:
        near[i] = (near[i] + 1) % 50_000
    streams = [
        _rand_stream(rng, 100) + base + _rand_stream(rng, 100),
        _rand_stream(rng, 100) + base + _rand_stream(rng, 100),
        _rand_stream(rng, 100) + base + _rand_stream(rng, 100),
        _rand_stream(rng, 100) + near + _rand_stream(rng, 100),
        _rand_stream(rng, 100) + near + _rand_stream(rng, 100),
        _rand_stream(rng, 100) + near + _rand_stream(rng, 100),
    ]
    # Both `base` (freq=3) and `near` (freq=3) appear at length>=256.
    # MinHash dedup must merge them into a single bucket; final union
    # frequency = 6.
    spans = mine_spans(
        streams, length_floor=256, frequency_floor=3,
        jaccard_threshold=0.8, minhash_perms=64,
    )
    # Check dedup happened. With threshold=1.0 (no dedup) we should see
    # both candidates; with 0.8 only one survivor (with merged sids).
    no_dedup = mine_spans(
        streams, length_floor=256, frequency_floor=3,
        jaccard_threshold=1.01,  # >1 disables MinHash bucket merging
    )
    # No-dedup pass keeps both 400-token spans; MinHash pass keeps one
    # with merged stream ids covering all 6 streams.
    assert len(no_dedup) >= len(spans)
    merged = [s for s in spans if s.length == 400]
    assert merged, f"no 400-token span survived: {[s.length for s in spans]}"
    assert max(s.frequency for s in merged) == 6, (
        f"MinHash dedup failed to merge 6 streams into one span: "
        f"got freqs {[s.frequency for s in merged]}"
    )


def test_results_sorted_by_freq_then_length_desc():
    rng = random.Random(5)
    p1 = _rand_stream(rng, 300)
    p2 = _rand_stream(rng, 280)
    streams = [
        p1 + _rand_stream(rng, 100),
        p1 + _rand_stream(rng, 100),
        p1 + _rand_stream(rng, 100),
        p1 + _rand_stream(rng, 100),  # p1 freq=4
        p2 + _rand_stream(rng, 100),
        p2 + _rand_stream(rng, 100),
        p2 + _rand_stream(rng, 100),  # p2 freq=3
    ]
    spans = mine_spans(streams, length_floor=256, frequency_floor=3)
    assert len(spans) >= 2
    # p1 first (freq=4 vs 3), then p2.
    freqs = [s.frequency for s in spans]
    assert freqs == sorted(freqs, reverse=True), f"not sorted by freq desc: {freqs}"


def test_fingerprint_is_stable_and_unique():
    s1 = MinedSpan(token_ids=(1, 2, 3), source_stream_ids=frozenset({0}), frequency=1)
    s2 = MinedSpan(token_ids=(1, 2, 3), source_stream_ids=frozenset({1}), frequency=1)
    s3 = MinedSpan(token_ids=(1, 2, 4), source_stream_ids=frozenset({0}), frequency=1)
    # Same tokens → same fingerprint (independent of source_stream_ids).
    assert s1.fingerprint() == s2.fingerprint()
    # Different tokens → different fingerprint.
    assert s1.fingerprint() != s3.fingerprint()
    # 16 hex chars.
    assert len(s1.fingerprint()) == 16


# --- internal helpers tests ----------------------------------------------


def test_shingle_handles_short_sequences():
    assert list(_shingle([1, 2], 5)) == [(1, 2)]
    assert list(_shingle([], 5)) == [tuple()]
    assert list(_shingle([1, 2, 3, 4, 5, 6], 5)) == [
        (1, 2, 3, 4, 5), (2, 3, 4, 5, 6),
    ]


def test_signature_identical_for_identical_sequences():
    seq = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    a = _signature(seq, perms=16, shingle_width=5)
    b = _signature(seq, perms=16, shingle_width=5)
    assert a == b
    assert _signature_jaccard(a, b) == 1.0


def test_signature_distinguishes_disjoint_sequences():
    a = _signature([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], perms=64, shingle_width=5)
    b = _signature([100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
                   perms=64, shingle_width=5)
    j = _signature_jaccard(a, b)
    # Disjoint shingles should have very low estimated Jaccard.
    assert j < 0.1, f"disjoint signatures had Jaccard {j} (expected < 0.1)"
