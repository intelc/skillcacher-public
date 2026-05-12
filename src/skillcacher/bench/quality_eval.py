"""Quality-eval driver.

Replays every (capture, turn) pair in a corpus across one condition (one
warm pod), N samples per turn, writes a parquet of generated outputs.

Run three times — once per condition (no_cache / prefix_cache /
cacheblend) — for the full §1 + §2 input. The CLI subcommand
(`bench/cli.py quality-eval`) handles the per-condition pod orchestration
via the existing ConditionLifecycle; this module is the single-condition
inner loop."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

import pyarrow as pa
import pyarrow.parquet as pq

from skillcacher.bench.output_capture import Generation, replay_with_output_capture
from skillcacher.proxy.trace_store import TraceStore

log = logging.getLogger("skillcacher.quality_eval")


def iter_replay_corpus(
    corpus_root: Path,
    *,
    capture_filter: set[str] | None = None,
) -> Iterator[tuple[str, int, dict]]:
    """Walk corpus_root for every traces.sqlite and yield (capture_id,
    turn_index, request_body).

    capture_id is the relative path from corpus_root to the capture dir,
    slash-joined — e.g. "swebench_verified/pylint-dev__pylint-7080".
    turn_index is the 0-based ts_start ordering within that capture.

    `capture_filter`, if supplied, is matched at the directory walk so
    captures outside the set are NEVER opened — important because the harness
    swebench_verified fixtures store absolute parquet paths in their
    SQLite that point at a _raw/ staging dir that no longer exists, so
    eagerly opening those captures crashes with FileNotFoundError.

    Captures whose parquet/sqlite reads fail (e.g., orphaned absolute
    paths) are logged + skipped; the eval continues. Same for malformed
    request_body_json rows."""
    for sqlite_path in sorted(corpus_root.glob("**/traces.sqlite")):
        capture_dir = sqlite_path.parent
        rel_parts = capture_dir.relative_to(corpus_root).parts
        capture_id = "/".join(rel_parts) if rel_parts else capture_dir.name
        if capture_filter is not None and capture_id not in capture_filter:
            continue
        store = TraceStore(capture_dir)
        try:
            records = list(store.read_all())
        except (FileNotFoundError, OSError) as e:
            log.warning(
                "capture=%s: trace_store read failed (%s); skipping capture",
                capture_id, e,
            )
            continue
        for turn_index, rec in enumerate(records):
            try:
                body = json.loads(rec.request_body_json)
            except (json.JSONDecodeError, TypeError) as e:
                log.warning(
                    "capture=%s turn=%d: request_body_json parse failed (%s); skipping",
                    capture_id, turn_index, e,
                )
                continue
            yield capture_id, turn_index, body


def _row_for(
    capture_id: str,
    turn_index: int,
    condition_name: str,
    sample_index: int,
    temperature: float,
    gen: Generation,
) -> dict:
    return {
        "capture_id": capture_id,
        "turn_index": turn_index,
        "condition": condition_name,
        "sample_index": sample_index,
        "temperature": temperature,
        "text": gen.text,
        "stop_reason": gen.stop_reason,
        "input_tokens": gen.input_tokens,
        "output_tokens": gen.output_tokens,
        "response_id": gen.response_id,
        "ttft_ms": gen.ttft_ms,
        "content_blocks_json": json.dumps(gen.content_blocks),
    }


def _row_for_failure(
    capture_id: str,
    turn_index: int,
    condition_name: str,
    sample_index: int,
    temperature: float,
    error: str,
) -> dict:
    return {
        "capture_id": capture_id,
        "turn_index": turn_index,
        "condition": condition_name,
        "sample_index": sample_index,
        "temperature": temperature,
        "text": "",
        "stop_reason": f"replay_error:{error[:200]}",
        "input_tokens": 0,
        "output_tokens": 0,
        "response_id": "",
        "ttft_ms": 0.0,
        "content_blocks_json": "[]",
    }


async def run_quality_eval(
    corpus_root: Path,
    *,
    condition_name: str,
    samples: int,
    temperature: float,
    out_dir: Path,
    settings,
    capture_filter: set[str] | None = None,
    replay_fn=replay_with_output_capture,
) -> Path:
    """Replay the corpus under one condition and write results to parquet.

    `capture_filter`, if supplied, restricts to captures whose capture_id
    is in the set. Used to subset (e.g., §2 picks 12 of 22 captures).

    `replay_fn` is injectable for tests."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for capture_id, turn_index, body in iter_replay_corpus(
        corpus_root, capture_filter=capture_filter,
    ):
        for sample_index in range(samples):
            try:
                gen = await replay_fn(body, settings, temperature=temperature)
                rows.append(_row_for(capture_id, turn_index, condition_name,
                                     sample_index, temperature, gen))
            except Exception as e:
                log.exception(
                    "replay failed: capture=%s turn=%d sample=%d",
                    capture_id, turn_index, sample_index,
                )
                rows.append(_row_for_failure(capture_id, turn_index, condition_name,
                                             sample_index, temperature, str(e)))

    out_path = out_dir / f"{condition_name}.parquet"
    if rows:
        pq.write_table(pa.Table.from_pylist(rows), out_path)
    return out_path
