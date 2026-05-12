"""Permuted-extended Layer 2 — extended-30 fixture builder.

30 turns of mixed-permutation MTRAG-Cloud requests over the same 5 passages.
Cacheblend-prowess cross-reference at scale: validates our cacheblend
implementation tracks the published MTRAG baseline across 30 sequential
requests, on a published-dataset workload readers can recognize. Where
mtrag_permuted_full_5 establishes correctness, extended_30 establishes
durability.

Builds on bench/permuted_fixture.py's full_5 generator — same passage shape,
same question pairing, just N=30 with question cycling when the cloud
corpus has fewer than 30 distinct questions available."""
from __future__ import annotations

import json
import random
from pathlib import Path

from skillcacher.bench.permuted_fixture import (
    Passage, build_body, _segment_sha256s,
)


N_REQUESTS = 30
N_PASSAGES = 5


def generate_extended_30(
    library: list[Passage], questions: list[str], *, seed: int
) -> list[dict]:
    """30 requests, same 5 passages from library[:5] in every request, full
    random shuffle per request, paired with questions (cycled if fewer than
    30 distinct).

    With 5 passages there are 5! = 120 distinct orderings, plenty for 30
    deduped reqs. Question diversity doesn't gate cacheblend behavior — what
    matters is the passage segments repeat across requests at varying
    positions."""
    if len(library) < N_PASSAGES:
        raise ValueError(f"need >={N_PASSAGES} library passages, got {len(library)}")
    if not questions:
        raise ValueError("need at least 1 question")

    base = list(library[:N_PASSAGES])
    bodies: list[dict] = []
    used_orders: set[tuple[str, ...]] = set()
    rng = random.Random(seed)
    while len(bodies) < N_REQUESTS:
        passages = list(base)
        rng.shuffle(passages)
        order = tuple(p.sha256_of_text for p in passages)
        if order in used_orders:
            continue
        used_orders.add(order)
        q = questions[len(bodies) % len(questions)]
        bodies.append(build_body(passages, q))
    return bodies


def write_extended_30_fixture(
    out_dir: Path, *, library: list[Passage], questions: list[str], seed: int
) -> None:
    """Write 30 raw_requests/*.json + manifest.json into out_dir. Wipes
    stale fixtures on re-run so the manifest stays in sync."""
    bodies = generate_extended_30(library, questions, seed=seed)
    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw_requests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for stale in raw_dir.glob("*.json"):
        stale.unlink()

    requests_meta: list[dict] = []
    for idx, body in enumerate(bodies):
        path = raw_dir / f"{idx:06d}.json"
        path.write_text(json.dumps(body, indent=2, sort_keys=True))
        requests_meta.append({
            "idx": idx,
            "expected_segment_hashes": _segment_sha256s(body),
        })

    manifest = {
        "mode": "extended_30",
        "seed": seed,
        "library": [
            {"title": p.title, "sha256_of_text": p.sha256_of_text}
            for p in library[:N_PASSAGES]
        ],
        "requests": requests_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
