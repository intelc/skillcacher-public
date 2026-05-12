"""statistical span miner.

Mines token-stream corpora (the per-request token parquets emitted by the
trace store) for repeated subsequences that occur in ≥ ``frequency_floor``
distinct request streams and are at least ``length_floor`` tokens long.

Algorithm: classic suffix-array + LCP "maximal repeats across documents"
formulation. Pure-Python; the corpora we care about (Layer 1 SWE-V +
post_compact + skill_invocation) are <1M tokens combined, well inside
Python's reach. If profile flips this hot, swap the SA construction for
pydivsufsort.

Dedup is two-stage:
1. Exact prefix-dedup: if a candidate's tokens are a strict prefix of
   another's, drop the shorter one.
2. MinHash near-dup: shingle each candidate at width 5, take a 64-perm
   signature, bucket by signature similarity ≥ ``jaccard_threshold``;
   keep the longest representative per bucket.

Output: ``MinedSpan(token_ids, source_stream_ids, frequency)``. Each entry
is ready to be ``span_registry.register(..., source="statistical")``-ed.

Why suffix array + LCP, not k-grams: k-grams require committing to a fixed
k upfront; SA+LCP discovers maximal repeats of variable length in one
pass, which matches the "find the natural prefix that recurs" shape we
want.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, Sequence


# Sentinel tokens are encoded as negative ints (vLLM/HF token IDs are
# non-negative), so they are guaranteed unique vs real tokens. Each
# stream gets its own sentinel so SA never coalesces suffixes across
# stream boundaries beyond the boundary character itself.
_SENTINEL_BASE = -(2**31)


@dataclass(frozen=True)
class MinedSpan:
    """One span discovered by the miner.

    `token_ids` is the canonical token sequence (taken from the first
    stream where it occurs). `source_stream_ids` is the set of stream
    indices in which this span was observed. `frequency` is
    `len(source_stream_ids)`."""
    token_ids: tuple[int, ...]
    source_stream_ids: frozenset[int]
    frequency: int

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def fingerprint(self) -> str:
        """Stable id for the span. SHA-256 of the token bytes, 16 hex
        chars. Used as the span_id when registering."""
        h = hashlib.sha256()
        for t in self.token_ids:
            h.update(t.to_bytes(4, "big", signed=False))
        return h.hexdigest()[:16]


def mine_spans(
    streams: Sequence[Sequence[int]],
    *,
    length_floor: int = 256,
    frequency_floor: int = 3,
    jaccard_threshold: float = 0.8,
    minhash_perms: int = 64,
    shingle_width: int = 5,
) -> list[MinedSpan]:
    """Find token sequences of length ≥ ``length_floor`` that appear in
    ≥ ``frequency_floor`` distinct streams. Dedup near-duplicates at
    Jaccard ≥ ``jaccard_threshold`` (over k=``shingle_width`` shingles).

    Returns spans sorted by (frequency desc, length desc) — most-impactful
    first, so callers can take the top-N if they want a budget cap."""
    if not streams or frequency_floor < 2:
        return []

    # 1. Build the concatenated token sequence + a parallel stream-id
    # array so we can look up the originating stream for any position.
    concat: list[int] = []
    stream_of_pos: list[int] = []
    stream_starts: list[int] = []
    for sid, stream in enumerate(streams):
        if len(stream) == 0:
            stream_starts.append(len(concat))
            continue
        stream_starts.append(len(concat))
        concat.extend(stream)
        stream_of_pos.extend([sid] * len(stream))
        # Append a unique sentinel so suffixes don't run past stream
        # boundaries. The sentinel itself gets a stream_of_pos entry too,
        # but it never appears in a maximal repeat that hits length_floor.
        concat.append(_SENTINEL_BASE - sid)
        stream_of_pos.append(sid)

    n = len(concat)
    if n < length_floor * frequency_floor:
        return []

    # 2. Build suffix array via prefix-doubling. The naive
    # `sorted(range(n), key=lambda i: concat[i:])` does O(n²) work
    # because each comparison walks the full common prefix; on a 250K-
    # token CC corpus that exceeds 10⁹ comparisons and Python hangs.
    # Prefix-doubling sorts by rank-of-(suffix[i], suffix[i+k]) tuples,
    # doubling k each round; total work is O(n log² n) with O(n) memory.
    sa = _build_suffix_array(concat)

    # 3. Build LCP array (Kasai algorithm — O(n)).
    rank = [0] * n
    for i, s in enumerate(sa):
        rank[s] = i
    lcp = [0] * n
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and concat[i + h] == concat[j + h]:
                h += 1
            lcp[rank[i]] = h
            if h > 0:
                h -= 1

    # 4. Walk LCP with a monotonic stack to enumerate maximal repeats.
    #    Standard algorithm: for each entry on the stack (h, left_idx),
    #    SA[left_idx-1 .. i-1] share at least `h` tokens. When the
    #    current LCP drops below the stack top, pop and emit candidates.
    #    This captures the NESTED case: a 350-token block of freq=3 is
    #    surrounded by a 300-token block of freq=5; both are emitted.
    candidates: list[MinedSpan] = []
    seen_fps: set[str] = set()

    def _emit(h: int, left_idx: int, right_idx: int) -> None:
        """SA[left_idx - 1 .. right_idx - 1] share at least `h` tokens.
        Emit a candidate if length and frequency floors are met and we
        haven't seen this exact fingerprint before."""
        if h < length_floor:
            return
        sids = {stream_of_pos[sa[j]] for j in range(left_idx - 1, right_idx)}
        if len(sids) < frequency_floor:
            return
        start = sa[left_idx - 1]
        token_ids = tuple(concat[start : start + h])
        if any(t < 0 for t in token_ids):
            return
        cand = MinedSpan(
            token_ids=token_ids,
            source_stream_ids=frozenset(sids),
            frequency=len(sids),
        )
        fp = cand.fingerprint()
        if fp in seen_fps:
            return
        seen_fps.add(fp)
        candidates.append(cand)

    # Stack entries: (lcp_height, left_idx) where left_idx is the SA index
    # such that all LCP values at positions [left_idx .. current-1] are
    # ≥ lcp_height. Sentinel 0 at the bottom so we never under-pop.
    stack: list[tuple[int, int]] = [(0, 0)]
    for i in range(1, n + 1):
        cur = lcp[i] if i < n else 0
        last_left = i
        while stack[-1][0] > cur:
            h_pop, left_pop = stack.pop()
            _emit(h_pop, left_pop, i)
            last_left = left_pop
        if stack[-1][0] < cur:
            stack.append((cur, last_left))

    # 5. Exact-prefix dedup: if span A's tokens are a prefix of span B's
    # AND |A| < |B| AND they share ≥1 stream, drop A (B is strictly more
    # informative). Sort by length desc so the long ones win.
    candidates.sort(key=lambda s: -s.length)
    kept: list[MinedSpan] = []
    for cand in candidates:
        is_prefix_of_kept = False
        for k in kept:
            if (
                k.length > cand.length
                and k.token_ids[: cand.length] == cand.token_ids
            ):
                is_prefix_of_kept = True
                break
        if not is_prefix_of_kept:
            kept.append(cand)

    # 6. MinHash near-dup: shingle width 5, 64-perm signature, bucket
    # pairs whose signature Jaccard ≥ jaccard_threshold; keep the longest
    # representative.
    if jaccard_threshold < 1.0 and len(kept) > 1:
        kept = _minhash_dedup(
            kept,
            perms=minhash_perms,
            shingle_width=shingle_width,
            threshold=jaccard_threshold,
        )

    # 7. Final sort: most-impactful first.
    kept.sort(key=lambda s: (-s.frequency, -s.length))
    return kept


# --- Suffix array construction -------------------------------------------


def _build_suffix_array(s: Sequence[int]) -> list[int]:
    """Prefix-doubling SA over an integer sequence. Returns the sorted
    suffix index list. O(n log² n) time, O(n) memory.

    The algorithm: rank[i] is the lex rank of suffix i when only the
    first `k` characters are considered. Initially k=1 and ranks come
    from a stable sort on `s[i]`. Each round doubles k by sorting on the
    composite key (rank[i], rank[i+k]) — equivalent to comparing the
    first 2k characters because both halves are length-k prefixes whose
    ranks are already known. Terminates as soon as all ranks are
    unique."""
    n = len(s)
    if n == 0:
        return []
    sa = sorted(range(n), key=lambda i: s[i])
    rank = [0] * n
    rank[sa[0]] = 0
    for i in range(1, n):
        rank[sa[i]] = rank[sa[i - 1]] + (1 if s[sa[i]] != s[sa[i - 1]] else 0)
    k = 1
    while k < n:
        def key(i: int, _k: int = k, _r: list[int] = rank, _n: int = n) -> tuple[int, int]:
            return (_r[i], _r[i + _k] if i + _k < _n else -1)
        sa.sort(key=key)
        new_rank = [0] * n
        new_rank[sa[0]] = 0
        for i in range(1, n):
            new_rank[sa[i]] = new_rank[sa[i - 1]] + (
                1 if key(sa[i]) != key(sa[i - 1]) else 0
            )
        rank = new_rank
        if rank[sa[-1]] == n - 1:
            break
        k *= 2
    return sa


# --- MinHash dedup --------------------------------------------------------


def _shingle(tokens: Sequence[int], k: int) -> Iterable[tuple[int, ...]]:
    if len(tokens) < k:
        yield tuple(tokens)
        return
    for i in range(len(tokens) - k + 1):
        yield tuple(tokens[i : i + k])


def _hash_shingle(s: tuple[int, ...], seed: int) -> int:
    """64-bit hash via blake2b keyed by the perm seed. Fast enough for
    the corpus sizes we expect; switch to xxhash if it becomes hot."""
    h = hashlib.blake2b(digest_size=8, key=seed.to_bytes(8, "big"))
    for t in s:
        h.update(t.to_bytes(4, "big", signed=False))
    return int.from_bytes(h.digest(), "big")


def _signature(tokens: Sequence[int], *, perms: int, shingle_width: int) -> tuple[int, ...]:
    shingles = list(_shingle(tokens, shingle_width))
    if not shingles:
        return tuple([0] * perms)
    sig = []
    for p in range(perms):
        sig.append(min(_hash_shingle(sh, p) for sh in shingles))
    return tuple(sig)


def _signature_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """MinHash signature Jaccard estimate: fraction of perm slots that
    agree."""
    if not a or len(a) != len(b):
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


def _minhash_dedup(
    spans: list[MinedSpan],
    *,
    perms: int,
    shingle_width: int,
    threshold: float,
) -> list[MinedSpan]:
    """Greedy MinHash near-dup merge: walk spans longest-first; for each
    candidate, attach to the first kept bucket whose signature Jaccard
    exceeds threshold; otherwise start a new bucket. Per bucket, the
    seeded longest is the representative.

    O(n²) worst case; fine for n in the hundreds. If the bucket count
    grows past ~1000, swap for an LSH index over band-hashes."""
    sigs = [
        _signature(s.token_ids, perms=perms, shingle_width=shingle_width)
        for s in spans
    ]
    bucket_repr_indices: list[int] = []
    span_to_bucket: list[int] = [-1] * len(spans)

    for i, sig in enumerate(sigs):
        attached = False
        for b_idx, repr_i in enumerate(bucket_repr_indices):
            if _signature_jaccard(sig, sigs[repr_i]) >= threshold:
                # Attach to this bucket. Repr stays as the longest seen
                # so far (we walk longest-first, so the first attached
                # one is the longest).
                span_to_bucket[i] = b_idx
                attached = True
                break
        if not attached:
            span_to_bucket[i] = len(bucket_repr_indices)
            bucket_repr_indices.append(i)

    # Per bucket, merge stream id sets (frequency reflects the union).
    merged: list[MinedSpan] = []
    for b_idx, repr_i in enumerate(bucket_repr_indices):
        union_sids: set[int] = set()
        for i, b in enumerate(span_to_bucket):
            if b == b_idx:
                union_sids |= spans[i].source_stream_ids
        merged.append(
            MinedSpan(
                token_ids=spans[repr_i].token_ids,
                source_stream_ids=frozenset(union_sids),
                frequency=len(union_sids),
            )
        )
    return merged
