"""`make bench` entry point — orchestrates per-condition pod lifecycle,
runs the workload through the proxy, joins chunk-level Lookup data into
decomposed hit rates, emits CSV + HTML report."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path

import click

# Configure root logging at module import. Without this, log.info() calls
# inside conditions.py — including the per-line oneshot_pod.sh streaming
# from `_run_streaming` — go to a logger with no handlers and disappear.
# Hit during 2026-05-07 Llama T36 v2: spent 8min staring at a blank log
# while the pod was provisioning, because the streaming was real but
# unreachable. Configure here, not at the conditions module level, so test
# imports don't slap a handler on the root logger.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)

from skillcacher.bench.checkpoint import CheckpointStore
from skillcacher.bench.conditions import Condition, ConditionLifecycle
from skillcacher.bench.metrics import aggregate
from skillcacher.bench.replay import replay_dir
from skillcacher.registry.span_registry import SpanRegistry
from skillcacher.settings import Settings
# NB: write_report is deferred-imported inside `run()` so this module can
# land before an upstream task ships report.py.


def _load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from .env into os.environ unless already set.

    Without this, bench runs invoked without `source .env` would lose
    RUNPOD_API_KEY (etc.) — and ConditionLifecycle's pod-cleanup DELETE
    would silently no-op, leaking GPU $$$ until manually cleaned up. Real
    incident on first T35 run; this is the fix."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = val


@click.group()
def main():
    """skillcacher bench harness."""


def _condition_list(value: str) -> list[Condition]:
    out = []
    for name in value.split(","):
        name = name.strip()
        if name:
            out.append(Condition(name))
    return out


@main.command()
@click.option("--workload", required=True, help="workload name (e.g. swebench_lite_20)")
@click.option("--conditions", default="no_cache,prefix_cache,cacheblend",
              help="comma-separated list of conditions to run sequentially")
@click.option("--fixture-set", default="swebench_lite",
              help="fixture set tag (used in output paths to keep multiple sets separate)")
@click.option("--out", default=Path("benchmark/results"), type=click.Path(path_type=Path))
def run(workload, conditions, fixture_set, out):
    """Run the workload across the listed conditions in sequence."""
    _load_env_file()  # ensure RUNPOD_API_KEY etc. are present for pod cleanup
    settings = Settings()
    out = Path(out)
    run_id = f"{workload}_{fixture_set}"
    run_root = out / run_id
    run_root.mkdir(parents=True, exist_ok=True)
    cs = CheckpointStore(run_root)

    asyncio.run(_run_all(settings, _condition_list(conditions), workload, fixture_set, cs))

    # Build per-condition aggregates. Two paths today:
    # 1. Trace-store path (replay_dir + aggregate) — produces decomposed
    #    skill/non-skill rates BUT requires `span_lookups` populated by the
    #    warmup pre-seed flow, which depends on lmcache_shim (T34.5 phase 2).
    # 2. vllm.log scrape path (parse_vllm_log_for_hits + aggregate_log_hits) —
    #    produces overall hit_rate from the log lines `LMCache hit tokens: N`.
    #    Always populated as long as ConditionLifecycle SCP'd vllm.log out
    #    before pod delete (followup #6).
    # We prefer #2's overall hit count when available, fall back to #1.
    from skillcacher.bench.report import write_report
    from skillcacher.bench.log_metrics import (
        parse_vllm_log_for_hits, aggregate_log_hits,
    )
    span_id_to_tag = _build_span_id_to_tag(settings)
    per_condition_agg = {}
    for cond_name in [c.name for c in _condition_list(conditions)]:
        log_path = run_root / cond_name / "vllm.log"
        log_hits = parse_vllm_log_for_hits(log_path)
        if log_hits:
            per_condition_agg[cond_name] = aggregate_log_hits(log_hits)
            click.echo(
                f"  {cond_name}: {len(log_hits)} requests, "
                f"hit={per_condition_agg[cond_name].total_hit_tokens} / "
                f"total={per_condition_agg[cond_name].prompt_token_count} "
                f"({per_condition_agg[cond_name].overall_hit_rate:.1%})"
            )
        else:
            rates = replay_dir(run_root / cond_name / "traces", span_id_to_tag=span_id_to_tag)
            per_condition_agg[cond_name] = aggregate(rates)
            click.echo(f"  {cond_name}: vllm.log absent/empty → fell back to trace-store path")
    write_report(run_root, workload=workload, fixture_set=fixture_set, per_condition=per_condition_agg)
    click.echo(f"wrote report to {run_root / 'report.html'}")


async def _run_all(settings, conditions, workload, fixture_set, cs):
    import os
    for cond in conditions:
        # Per-condition SKILLCACHER_TRACE_DIR aligned with where replay_dir
        # reads — `run_root/<cond>/traces`. Set BEFORE entering the lifecycle
        # because the local proxy spawned inside reads env at startup.
        # SKILLCACHER_BACKEND_URL is set by ConditionLifecycle itself from
        # the pod's PROXY_URL — no need to plumb it here.
        os.environ["SKILLCACHER_TRACE_DIR"] = str(cs.run_root / cond.name / "traces")
        # log_dump_path: ConditionLifecycle SCPs vllm.log here BEFORE deleting
        # the pod. Required for hit-rate metrics on the cu12 image (followup #6).
        log_dump_path = cs.run_root / cond.name / "vllm.log"
        async with ConditionLifecycle(cond, log_dump_path=log_dump_path):
            await _replay_workload(settings, workload, fixture_set, cs, cond)


async def _replay_workload(settings, workload, fixture_set, cs, cond):
    """Replay a captured fixture against the live proxy for one condition.
    Resumes from the next-index checkpoint."""
    from skillcacher.bench.workload import iter_requests, send_request
    start = cs.next_index(cond.name)
    for idx, req_body in iter_requests(fixture_set, workload, start_at=start):
        record = await send_request(req_body, settings)
        cs.write_request(cond.name, idx, record)


def _build_span_id_to_tag(settings: Settings) -> dict[str, str]:
    reg = SpanRegistry(settings.span_registry_path)
    out = {}
    # All registered spans in the harness are skill_body (pre-seeded by warmup);
    # tool_def / system_static spans are not registered (deferred).
    for span in reg.spans_referenced_by(skill_ids=_all_skill_ids(reg)):
        out[span.span_id] = "skill_body"
    return out


def _all_skill_ids(reg: SpanRegistry) -> list[str]:
    import sqlite3
    with sqlite3.connect(reg.db_path) as conn:
        rows = conn.execute("SELECT DISTINCT skill_id FROM spans").fetchall()
    return [r[0] for r in rows]


@main.command(name="quality-eval")
@click.option("--corpus-dir", required=True,
              type=click.Path(path_type=Path, exists=True, file_okay=False),
              help="Local dir containing the staged ClaudeCodeTrace dataset "
                   "(produced by scripts/download_claudecode_trace.py)")
@click.option("--conditions", default="no_cache,prefix_cache,cacheblend",
              help="Comma-separated list of conditions to run sequentially")
@click.option("--samples", default=1, type=int,
              help="Samples per (capture, turn, condition)")
@click.option("--temperature", default=0.0, type=float,
              help="Decoding temperature; overrides whatever the captured "
                   "request body specified")
@click.option("--out", default=Path("benchmark/results/plan5_quality"),
              type=click.Path(path_type=Path),
              help="Output dir for per-condition parquets")
@click.option("--capture-filter", default=None,
              help="Optional comma-separated capture_id list to restrict the "
                   "replay corpus (used by §2's 12-of-22 subset)")
def quality_eval(corpus_dir, conditions, samples, temperature, out, capture_filter):
    """replay corpus across conditions, capture model outputs."""
    _load_env_file()
    settings = Settings()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cf = None
    if capture_filter:
        cf = {c.strip() for c in capture_filter.split(",") if c.strip()}
    asyncio.run(_run_quality_eval_all(
        settings, _condition_list(conditions), Path(corpus_dir),
        samples, temperature, out, cf,
    ))


async def _run_quality_eval_all(settings, conditions, corpus_dir, samples,
                                temperature, out, capture_filter):
    from skillcacher.bench.quality_eval import run_quality_eval
    for cond in conditions:
        cond_traces = out / cond.name / "traces"
        cond_traces.parent.mkdir(parents=True, exist_ok=True)
        os.environ["SKILLCACHER_TRACE_DIR"] = str(cond_traces)
        log_dump_path = out / cond.name / "vllm.log"
        async with ConditionLifecycle(cond, log_dump_path=log_dump_path):
            await run_quality_eval(
                corpus_dir,
                condition_name=cond.name,
                samples=samples,
                temperature=temperature,
                out_dir=out,
                settings=settings,
                capture_filter=capture_filter,
            )
        click.echo(f"  {cond.name}: wrote {out / f'{cond.name}.parquet'}")


if __name__ == "__main__":
    main()
