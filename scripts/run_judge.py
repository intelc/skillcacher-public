"""Driver for the §5.3 LLM judge: judge cacheblend vs no_cache.

Reads `benchmark/results/<run>/{no_cache,cacheblend}.parquet`,
joins per (capture_id, turn_index), reconstructs each turn's task prompt
from the original captured request body, calls the judge for each pair
with random A/B position, writes a preferences CSV + a summary stats line.

Skips pairs where either side failed (`stop_reason: replay_error:*`).
Skips bit-identical pairs (no informative judgment to make) by default —
pass --include-identical to judge them too as a control."""
from __future__ import annotations

import csv
import json
import logging
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import click
import pyarrow.parquet as pq

# scripts/ isn't a package; import the project via path
_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "src"))

from skillcacher.bench.output_capture import Generation
from skillcacher.bench.output_compare import canonicalize, token_identity_rate
from skillcacher.judge.prompt import JudgePair, render_call
from skillcacher.judge.sonnet_judge import call_judge_with_retry, make_client
from skillcacher.proxy.trace_store import TraceStore

log = logging.getLogger("skillcacher.run_judge")


def _load_condition(parquet_path: Path) -> dict[tuple[str, int], dict]:
    """Load one condition's parquet, indexed by (capture_id, turn_index)."""
    rows = pq.read_table(parquet_path).to_pylist()
    out: dict[tuple[str, int], dict] = {}
    for r in rows:
        if r["stop_reason"].startswith("replay_error"):
            continue
        if r["sample_index"] != 0:  # §1 ran samples=1; defensive
            continue
        out[(r["capture_id"], r["turn_index"])] = r
    return out


def _gen_from(row: dict) -> Generation:
    blocks = json.loads(row["content_blocks_json"]) if row["content_blocks_json"] else []
    return Generation(
        text=row["text"], content_blocks=blocks,
        stop_reason=row["stop_reason"],
        input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
        response_id=row["response_id"], ttft_ms=row["ttft_ms"],
    )


def _extract_task_prompt(body: dict) -> str:
    """Best-effort: take the last user message's text content as the
    task prompt the agent was responding to. CC bodies can have
    `messages[i].content` as a string OR list of {type, text/...}."""
    msgs = body.get("messages") or []
    for m in reversed(msgs):
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            return c[:4000]
        if isinstance(c, list):
            parts = [b.get("text", "") for b in c if b.get("type") == "text"]
            joined = "\n".join(p for p in parts if p)
            if joined:
                return joined[:4000]
    return ""


def _load_task_prompts(corpus_root: Path,
                       capture_ids: set[str]) -> dict[tuple[str, int], str]:
    """For each (capture_id, turn_index) in scope, fetch the task prompt
    from the original TraceStore."""
    out: dict[tuple[str, int], str] = {}
    for capture_id in capture_ids:
        capture_dir = corpus_root / capture_id
        if not (capture_dir / "traces.sqlite").exists():
            log.warning("capture=%s: traces.sqlite missing, skipping prompt extraction", capture_id)
            continue
        store = TraceStore(capture_dir)
        try:
            records = list(store.read_all())
        except Exception as e:
            log.warning("capture=%s: trace_store read failed (%s); skipping prompt extraction", capture_id, e)
            continue
        for turn_index, rec in enumerate(records):
            try:
                body = json.loads(rec.request_body_json)
            except (json.JSONDecodeError, TypeError):
                continue
            out[(capture_id, turn_index)] = _extract_task_prompt(body)
    return out


def _summary(rows: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {"n": 0}
    cb = sum(1 for r in rows if r["prefers"] == "cacheblend")
    nc = sum(1 for r in rows if r["prefers"] == "no_cache")
    eq = sum(1 for r in rows if r["prefers"] == "equivalent")
    un = sum(1 for r in rows if r["prefers"] == "unparseable")
    # cacheblend non-inferior = wins-or-ties / parseable-judgments
    parseable = cb + nc + eq
    non_inferior = (cb + eq) / parseable if parseable else 0.0
    # Wilson 95% CI on cacheblend preference rate (cb / parseable)
    p = cb / parseable if parseable else 0.0
    z = 1.96
    if parseable:
        denom = 1 + z * z / parseable
        center = (p + z * z / (2 * parseable)) / denom
        rad = (z / denom) * ((p * (1 - p) / parseable + z * z / (4 * parseable * parseable)) ** 0.5)
        lo, hi = max(0.0, center - rad), min(1.0, center + rad)
    else:
        lo, hi = 0.0, 0.0
    return {
        "n": n,
        "n_parseable": parseable,
        "n_unparseable": un,
        "cacheblend_wins": cb,
        "no_cache_wins": nc,
        "equivalent": eq,
        "cacheblend_preference_rate": round(p, 4),
        "wilson95_lo": round(lo, 4),
        "wilson95_hi": round(hi, 4),
        "cacheblend_non_inferior_rate": round(non_inferior, 4),
    }


@click.command()
@click.option("--corpus-dir", default="tests/fixtures/claude_code_real",
              type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--results-dir", default="benchmark/results/plan5_quality_section1",
              type=click.Path(path_type=Path, exists=True, file_okay=False))
@click.option("--out", default="benchmark/results/plan5_quality_section1/judge_preferences.csv",
              type=click.Path(path_type=Path))
@click.option("--seed", default=42, type=int, help="Random seed for A/B position assignment")
@click.option("--include-identical/--skip-identical", default=False,
              help="Include bit-identical pairs as controls (judge should call EQUIVALENT)")
@click.option("--limit", default=None, type=int, help="Limit number of pairs (for cost control)")
@click.option("--model", default="claude-sonnet-4-6", help="Judge model")
def main(corpus_dir, results_dir, out, seed, include_identical, limit, model):
    """Pairwise judge cacheblend vs no_cache outputs."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    nc = _load_condition(results_dir / "no_cache.parquet")
    cb = _load_condition(results_dir / "cacheblend.parquet")
    keys = sorted(set(nc.keys()) & set(cb.keys()))
    log.info("loaded %d no_cache + %d cacheblend rows; %d shared (capture, turn) pairs",
             len(nc), len(cb), len(keys))

    capture_ids = {k[0] for k in keys}
    task_prompts = _load_task_prompts(corpus_dir, capture_ids)
    log.info("extracted task prompts for %d (capture, turn) pairs", len(task_prompts))

    rng = random.Random(seed)
    client = make_client()
    rows: list[dict] = []
    skipped_identical = 0
    judged = 0
    for key in keys:
        capture_id, turn_index = key
        nc_gen = _gen_from(nc[key])
        cb_gen = _gen_from(cb[key])
        identity = token_identity_rate(nc_gen, cb_gen)
        if not include_identical and identity >= 0.9999:
            skipped_identical += 1
            rows.append({
                "capture_id": capture_id, "turn_index": turn_index,
                "identity": identity,
                "prefers": "equivalent_skipped",
                "label": "SKIPPED_IDENTICAL", "rationale": "outputs bit-identical, judge skipped",
                "position_a": "n/a", "raw_response": "",
            })
            continue
        if limit is not None and judged >= limit:
            log.info("hit --limit=%d, stopping", limit)
            break
        pair = JudgePair(
            capture_id=capture_id, turn_index=turn_index,
            cacheblend_text=canonicalize(cb_gen),
            no_cache_text=canonicalize(nc_gen),
            task_prompt=task_prompts.get(key, ""),
        )
        call = render_call(pair, rng=rng)
        result = call_judge_with_retry(call, client=client, model=model)
        judged += 1
        log.info("[%d/%d] %s turn=%d identity=%.4f → %s (%s)",
                 judged, len(keys), capture_id, turn_index, identity,
                 result.label, result.prefers)
        rows.append({
            "capture_id": capture_id, "turn_index": turn_index,
            "identity": round(identity, 4),
            "prefers": result.prefers, "label": result.label,
            "rationale": result.rationale, "position_a": result.position_a,
            "raw_response": result.raw_response[:500],
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else
                                ["capture_id", "turn_index", "identity", "prefers",
                                 "label", "rationale", "position_a", "raw_response"])
        writer.writeheader()
        writer.writerows(rows)
    log.info("wrote %d rows to %s (skipped %d bit-identical pairs, judged %d)",
             len(rows), out, skipped_identical, judged)
    summary = _summary([r for r in rows if r["prefers"] != "equivalent_skipped"])
    log.info("=== SUMMARY (excludes %d bit-identical pairs) ===", skipped_identical)
    for k, v in summary.items():
        log.info("  %s = %s", k, v)


if __name__ == "__main__":
    main()
