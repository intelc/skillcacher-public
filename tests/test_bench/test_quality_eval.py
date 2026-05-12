"""Tests for bench/quality_eval.py — corpus iteration + replay driver."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pyarrow.parquet as pq

from skillcacher.bench.output_capture import Generation
from skillcacher.bench.quality_eval import iter_replay_corpus, run_quality_eval
from skillcacher.proxy.trace_store import RequestRecord, TraceStore
from skillcacher.settings import Settings


def _make_capture(capture_dir: Path, *requests: dict) -> None:
    """Write a synthetic TraceStore at capture_dir with the given request bodies
    (one RequestRecord per body, ts_start ordered)."""
    store = TraceStore(capture_dir)
    store.init_schema()
    for i, body in enumerate(requests):
        store.write(RequestRecord(
            request_id=f"req_{capture_dir.name}_{i}",
            session_id="s1",
            ts_start=float(i),
            ts_end=float(i) + 0.5,
            prompt_tokens=[1, 2, 3],
            response_tokens=[],
            prompt_token_count=3,
            response_token_count=0,
            cache_read_tokens=0,
            cache_recompute_tokens=0,
            ttft_ms=100.0,
            request_body_json=json.dumps(body),
        ))


def test_iter_replay_corpus_walks_subdirs(tmp_path):
    # corpus/swebench_verified/task_a/, corpus/skill_invocation/x/
    a = tmp_path / "swebench_verified" / "task_a"
    b = tmp_path / "skill_invocation" / "x"
    _make_capture(a, {"messages": [{"role": "user", "content": "a1"}]},
                  {"messages": [{"role": "user", "content": "a2"}]})
    _make_capture(b, {"messages": [{"role": "user", "content": "b1"}]})

    yielded = list(iter_replay_corpus(tmp_path))
    capture_ids = {row[0] for row in yielded}
    assert capture_ids == {"swebench_verified/task_a", "skill_invocation/x"}
    # Both turns of capture_a, one of capture_b → 3 total
    assert len(yielded) == 3


def test_iter_replay_corpus_yields_in_turn_order(tmp_path):
    cap = tmp_path / "single"
    _make_capture(cap, {"q": "first"}, {"q": "second"}, {"q": "third"})
    yielded = list(iter_replay_corpus(tmp_path))
    # Same capture, three turns, in 0..2 order
    assert [row[1] for row in yielded] == [0, 1, 2]
    assert [row[2]["q"] for row in yielded] == ["first", "second", "third"]


def test_iter_replay_corpus_capture_filter_skips_unopened(tmp_path):
    # Create capture A with a valid TraceStore.
    a = tmp_path / "keep"
    _make_capture(a, {"q": "k"})
    # Create capture B with a corrupted SQLite that would crash on read_all().
    # Put a junk file at traces.sqlite — sqlite3.connect will succeed (creates
    # a new db), but the SCHEMA isn't there so SELECT will fail. We're really
    # asserting the FILTER prevents the open in the first place.
    b = tmp_path / "skip"
    b.mkdir()
    (b / "traces.sqlite").write_text("not a sqlite database, just bytes")

    yielded = list(iter_replay_corpus(tmp_path, capture_filter={"keep"}))
    assert {row[0] for row in yielded} == {"keep"}


def test_iter_replay_corpus_skips_capture_with_unreadable_parquet(tmp_path, caplog):
    # Capture A: valid.
    a = tmp_path / "good"
    _make_capture(a, {"q": "g"})
    # Capture B: SQLite present but its token_parquet_path points at a file
    # that doesn't exist (mirrors the the harness swebench_verified absolute-path
    # issue post-dir-move).
    b = tmp_path / "broken"
    b.mkdir()
    import sqlite3
    (b / "tokens").mkdir()
    db = b / "traces.sqlite"
    with sqlite3.connect(db) as conn:
        conn.executescript(TraceStore.SCHEMA)
        conn.execute(
            "INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("req_x", "s", 1.0, 2.0, 1, 0, 0, 0, None, None, None, 0, "[]",
             0.0, '{}', "/nonexistent/path/req_x.parquet", None),
        )
    yielded = list(iter_replay_corpus(tmp_path))
    # Good capture still yielded; broken one skipped.
    assert [row[0] for row in yielded] == ["good"]


def test_iter_replay_corpus_skips_malformed_request_body(tmp_path, caplog):
    cap = tmp_path / "bad"
    store = TraceStore(cap)
    store.init_schema()
    store.write(RequestRecord(
        request_id="req_bad",
        session_id="s1", ts_start=1.0, ts_end=2.0,
        prompt_tokens=[1], response_tokens=[],
        prompt_token_count=1, response_token_count=0,
        cache_read_tokens=0, cache_recompute_tokens=0,
        ttft_ms=0.0,
        request_body_json="{not json",  # malformed
    ))
    # Add a good one too — should still be yielded.
    store.write(RequestRecord(
        request_id="req_good",
        session_id="s1", ts_start=2.0, ts_end=3.0,
        prompt_tokens=[1], response_tokens=[],
        prompt_token_count=1, response_token_count=0,
        cache_read_tokens=0, cache_recompute_tokens=0,
        ttft_ms=0.0,
        request_body_json='{"ok": true}',
    ))
    yielded = list(iter_replay_corpus(tmp_path))
    assert len(yielded) == 1
    assert yielded[0][2] == {"ok": True}


def test_run_quality_eval_writes_parquet_with_expected_rows(tmp_path):
    corpus = tmp_path / "corpus"
    cap = corpus / "x"
    _make_capture(cap, {"messages": [{"role": "user", "content": "hi"}]},
                  {"messages": [{"role": "user", "content": "bye"}]})
    out_dir = tmp_path / "out"

    async def fake_replay(req_body, settings, *, temperature=None):
        return Generation(
            text=f"reply-to-{req_body['messages'][0]['content']}",
            content_blocks=[{"type": "text", "text": "x"}],
            stop_reason="end_turn",
            input_tokens=10, output_tokens=2, response_id="r",
            ttft_ms=1.0,
        )

    out = asyncio.run(run_quality_eval(
        corpus, condition_name="cacheblend", samples=2, temperature=0.0,
        out_dir=out_dir, settings=Settings(), replay_fn=fake_replay,
    ))
    assert out.exists()
    table = pq.read_table(out)
    df = table.to_pylist()
    # 2 turns × 2 samples = 4 rows
    assert len(df) == 4
    # All sample_index values present
    assert sorted({r["sample_index"] for r in df}) == [0, 1]
    # All turn_index values present
    assert sorted({r["turn_index"] for r in df}) == [0, 1]
    # Condition stamped on every row
    assert {r["condition"] for r in df} == {"cacheblend"}
    # Temperature stamped on every row
    assert {r["temperature"] for r in df} == {0.0}


def test_run_quality_eval_records_failure_rows_on_replay_error(tmp_path):
    corpus = tmp_path / "corpus"
    cap = corpus / "x"
    _make_capture(cap, {"messages": []})
    out_dir = tmp_path / "out"

    async def boom_replay(req_body, settings, *, temperature=None):
        raise RuntimeError("backend down")

    asyncio.run(run_quality_eval(
        corpus, condition_name="no_cache", samples=1, temperature=0.0,
        out_dir=out_dir, settings=Settings(), replay_fn=boom_replay,
    ))
    table = pq.read_table(out_dir / "no_cache.parquet")
    rows = table.to_pylist()
    assert len(rows) == 1
    assert rows[0]["stop_reason"].startswith("replay_error:")
    assert "backend down" in rows[0]["stop_reason"]


def test_run_quality_eval_capture_filter_subsets_corpus(tmp_path):
    corpus = tmp_path / "corpus"
    _make_capture(corpus / "keep", {"q": "k"})
    _make_capture(corpus / "skip", {"q": "s"})
    out_dir = tmp_path / "out"

    async def fake_replay(req_body, settings, *, temperature=None):
        return Generation(text="x", content_blocks=[], stop_reason="end_turn",
                          input_tokens=0, output_tokens=0, response_id="",
                          ttft_ms=0.0)

    asyncio.run(run_quality_eval(
        corpus, condition_name="c", samples=1, temperature=0.0,
        out_dir=out_dir, settings=Settings(), replay_fn=fake_replay,
        capture_filter={"keep"},
    ))
    rows = pq.read_table(out_dir / "c.parquet").to_pylist()
    assert {r["capture_id"] for r in rows} == {"keep"}
