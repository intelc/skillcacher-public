"""Build permuted-MTRAG fixtures for the discriminating-workload bench gate.

Two modes:
  - full_5: 5 reqs, all 5 passages from one MTRAG cloud record present in
    every request, only the order varies. Implementation-correctness gate.
  - partial_20: 20 reqs, library M=10 (2 records' contexts), each req picks
    K=5 with 4-of-5 always shared + 1 rotated. Realistic-RAG delta gate.

Body shape mirrors `scripts/dev/build_synthetic_swebench_lite_fixture.py`:
Anthropic chat-completions, passages-first user content, question last,
BLEND_SEP=' # # ' between every passage and between block and question.
"""
from __future__ import annotations

import hashlib
import json
import random
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

BLEND_SEP = " # # "  # MUST match LMCACHE_BLEND_SPECIAL_STR on the pod.
CLOUD_COLLECTION = "mt-rag-ibmcloud-elser-512-100-20240502"
SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "provided reference passages. Be concise."
)
MTRAG_URL = (
    "https://raw.githubusercontent.com/IBM/mt-rag-benchmark/main/"
    "mtrag-human/generation_tasks/reference%2BRAG.jsonl"
)


@dataclass(frozen=True)
class Passage:
    title: str
    text: str

    @property
    def formatted(self) -> str:
        return f"[{self.title}]\n{self.text}"

    @property
    def sha256_of_text(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()


def _iter_cloud(jsonl_path: Path, *, turns: tuple[str, ...] = ("1",)):
    """Yield records from the IBM cloud collection in the given turn(s).

    Default turns=("1",) preserves the strict "library = turn-1 records"
    invariant for `load_library`. `load_questions` passes turns=("1", "2")
    because the real MTRAG cloud subset only has ~16 distinct turn-1
    questions — not enough for partial_20's 20-question pool. Turn-2
    questions are still cloud-collection RAG queries; they just come
    from later in the same conversations."""
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d.get("Collection") != CLOUD_COLLECTION:
                continue
            if d.get("turn") not in turns:
                continue
            yield d


def load_library(jsonl_path: Path) -> list[Passage]:
    """Library = first 2 cloud turn-1 records each contributing 5 contexts."""
    records: list[dict] = []
    for d in _iter_cloud(jsonl_path):
        if len(d.get("contexts", [])) < 5:
            continue
        records.append(d)
        if len(records) == 2:
            break
    if len(records) < 2:
        raise RuntimeError(
            f"need 2 cloud turn-1 records with >=5 contexts each; found {len(records)}"
        )
    out: list[Passage] = []
    for r in records:
        for c in r["contexts"][:5]:
            title = c.get("title") or "Passage"
            out.append(Passage(title=title, text=c.get("text", "")))
    assert len(out) == 10, f"expected 10 library passages, got {len(out)}"
    return out


def load_questions(jsonl_path: Path, n: int) -> list[str]:
    """Distinct user questions from cloud turn-1 and turn-2 records."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for d in _iter_cloud(jsonl_path, turns=("1", "2")):
        msgs = d.get("input") or []
        if not msgs:
            continue
        q = msgs[-1].get("text", "").strip()
        if not q or q in seen_set:
            continue
        seen.append(q)
        seen_set.add(q)
        if len(seen) == n:
            break
    if len(seen) < n:
        raise RuntimeError(
            f"need {n} cloud turn-1 or turn-2 records with distinct questions; found {len(seen)}"
        )
    return seen


def download_mtrag(cache_path: Path) -> None:
    """Idempotent download of the public MTRAG file. No HF auth needed.

    If `cache_path` exists at all, we leave it alone — the caller is responsible
    for deleting a partial download manually before re-running. (Earlier versions
    re-downloaded when the cache was <1MB, which clobbered intentionally-small
    test fixtures pointed at via --cache.)"""
    if cache_path.exists():
        return
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(MTRAG_URL, headers={"User-Agent": "skillcacher-permuted/0.1"})
    with urllib.request.urlopen(req, timeout=60) as r:
        cache_path.write_bytes(r.read())


def build_body(passages: list[Passage], question: str,
               *, model: str = "Qwen/Qwen3-8B", max_tokens: int = 64) -> dict:
    """Anthropic chat-completions body, passages-first, question-last.

    The proxy translates Anthropic→OpenAI on the way to vllm; passage
    separators ride through verbatim so lmcache's SegmentTokenDatabase
    splits each passage into its own content-hashed segment."""
    block = BLEND_SEP + BLEND_SEP.join(p.formatted for p in passages)
    user_content = f"Reference passages:{block}{BLEND_SEP}Question: {question}"
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": [{"type": "text", "text": SYSTEM_PROMPT}],
        "messages": [{"role": "user", "content": user_content}],
    }


def generate_full_5(library: list[Passage], questions: list[str], *, seed: int) -> list[dict]:
    """Mode 1: 5 requests, all 5 passages from the FIRST record present in
    every request, **fully randomly shuffled per request**. Each request
    pairs with a distinct question. Implementation-correctness gate.

    Library is the full 10-passage set; we take the first 5 (which all came
    from one record, see load_library).

    Originally downgraded to a single-adjacent-swap rotation in `adbdf03` to
    work around an upstream lmcache 0.4.2 blender bug under full random
    shuffles. The bug is now fixed by the `LMCACHE_CACHEBLEND_DIMFIX` boot
    patch (see `scripts/dev/oneshot_pod.py:_cacheblend_patches`, upstream
    PR LMCache#3217), so this generator is restored to the spec's intended
    full-shuffle shape — which produces a stronger cacheblend > prefix_cache
    delta (prefix cache bails at the first reordered passage, which under a
    full shuffle is almost always position 0)."""
    if len(library) < 5:
        raise ValueError(f"need >=5 library passages, got {len(library)}")
    if len(questions) < 5:
        raise ValueError(f"need >=5 questions, got {len(questions)}")
    base = list(library[:5])
    bodies: list[dict] = []
    used_orders: set[tuple[str, ...]] = set()
    rng = random.Random(seed)
    while len(bodies) < 5:
        passages = list(base)
        rng.shuffle(passages)
        # Dedup on text-content hashes, not titles — real MTRAG cloud records
        # have title=None, so title-based keys collapse all permutations to a
        # single value and the loop spins forever.
        order = tuple(p.sha256_of_text for p in passages)
        if order in used_orders:
            continue
        used_orders.add(order)
        bodies.append(build_body(passages, questions[len(bodies)]))
    return bodies


def _segment_sha256s(body: dict) -> list[str]:
    """Sha256 each BLEND_SEP-delimited segment of the user content. Order
    matters — manifest stores the list in body order so reviewers can verify
    by re-deriving from the body."""
    content = body["messages"][0]["content"]
    parts = [p for p in content.split(BLEND_SEP) if p]
    return [hashlib.sha256(p.encode("utf-8")).hexdigest() for p in parts]


def write_fixture_set(
    out_dir: Path,
    *,
    mode: str,
    library: list[Passage],
    questions: list[str],
    seed: int,
) -> None:
    """Generate `mode`'s bodies and manifest into `out_dir`. Wipes any
    stale `raw_requests/*.json` first so re-runs don't accumulate cruft."""
    if mode == "full_5":
        bodies = generate_full_5(library, questions, seed=seed)
    elif mode == "partial_20":
        bodies = generate_partial_20(library, questions, seed=seed)
    else:
        raise ValueError(f"unknown mode: {mode}")

    out_dir = Path(out_dir)
    raw_dir = out_dir / "raw_requests"
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Wipe any stale fixtures from prior runs to keep manifest in sync.
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
        "mode": mode,
        "seed": seed,
        "library": [
            {"title": p.title, "sha256_of_text": p.sha256_of_text}
            for p in library
        ],
        "requests": requests_meta,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def generate_partial_20(library: list[Passage], questions: list[str], *, seed: int) -> list[dict]:
    """Mode 2: 20 requests, K=5 passages per request (4 always shared + 1
    rotated from outer 6), shuffled. Realistic-RAG delta gate.

    Library convention (from load_library):
      indices [0..3] = the first 4 passages → CORE set, always present.
      indices [4..9] = OUTER pool of 6, one rotates in per request.

    The '4 always shared' is the dominant invariant our test asserts on:
    intersection of passage-hashes across all 20 reqs >= 4."""
    if len(library) < 10:
        raise ValueError(f"need >=10 library passages, got {len(library)}")
    if len(questions) < 20:
        raise ValueError(f"need >=20 questions, got {len(questions)}")
    core = list(library[:4])
    outer = list(library[4:10])  # 6 passages
    bodies: list[dict] = []
    rng = random.Random(seed)
    for i in range(20):
        chosen = list(core) + [outer[i % len(outer)]]
        rng.shuffle(chosen)
        bodies.append(build_body(chosen, questions[i]))
    return bodies
