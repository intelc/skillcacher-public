"""SQLite (metadata) + Parquet (token streams + lookup data) trace store."""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

log = logging.getLogger("skillcacher.trace_store")


@dataclass
class ChunkHitMap:
    span_id: str
    chunk_hits: int
    total_chunks: int
    hit_tokens: int
    total_tokens: int


@dataclass
class RequestRecord:
    request_id: str
    session_id: str
    ts_start: float
    ts_end: float
    prompt_tokens: Sequence[int]
    response_tokens: Sequence[int]
    prompt_token_count: int
    response_token_count: int
    cache_read_tokens: int
    cache_recompute_tokens: int
    ttft_ms: float
    request_body_json: str
    span_tags: list[str] = field(default_factory=list)
    # additions
    span_lookups: list[ChunkHitMap] = field(default_factory=list)
    engine_total_hit_tokens: int | None = None
    engine_load_tokens: int | None = None
    tokens_recomputed_hkvd: int | None = None
    chunk_aligned_hit_tokens: int = 0
    invariant_violations: list[str] = field(default_factory=list)
    # Transient: structural span tag summary (kind, count) — recorded for debug
    _tag_summary: list = field(default_factory=list, repr=False)


class TraceStore:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS requests (
        request_id TEXT PRIMARY KEY,
        session_id TEXT NOT NULL,
        ts_start REAL NOT NULL,
        ts_end REAL NOT NULL,
        prompt_token_count INTEGER NOT NULL,
        response_token_count INTEGER NOT NULL,
        cache_read_tokens INTEGER NOT NULL,
        cache_recompute_tokens INTEGER NOT NULL,
        engine_total_hit_tokens INTEGER,
        engine_load_tokens INTEGER,
        tokens_recomputed_hkvd INTEGER,
        chunk_aligned_hit_tokens INTEGER NOT NULL DEFAULT 0,
        invariant_violations TEXT NOT NULL DEFAULT '[]',
        ttft_ms REAL NOT NULL,
        request_body_json TEXT NOT NULL,
        token_parquet_path TEXT NOT NULL,
        lookups_parquet_path TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_session ON requests(session_id);
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "tokens").mkdir(exist_ok=True)
        (self.root / "lookups").mkdir(exist_ok=True)
        self.db_path = self.root / "traces.sqlite"

    def init_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def write(self, rec: RequestRecord) -> None:
        # Invariant 1: tag length matches token length (when both nonempty)
        if rec.span_tags and len(rec.span_tags) != len(rec.prompt_tokens):
            raise ValueError(
                f"span_tags length {len(rec.span_tags)} != prompt_tokens length {len(rec.prompt_tokens)}"
            )
        # Invariant 2: no negative token ids
        for t in list(rec.prompt_tokens) + list(rec.response_tokens):
            if t < 0:
                raise ValueError(f"negative token id: {t}")
        # Invariant 3 (warning, not fatal): lookup sum <= engine total
        if rec.engine_total_hit_tokens is not None:
            lookup_sum = sum(l.hit_tokens for l in rec.span_lookups)
            if lookup_sum > rec.engine_total_hit_tokens:
                # Copy-on-write so re-calling write() with the same record
                # doesn't double-append the violation. Dedupe so an idempotent
                # retry with the same record produces the same violation set.
                if "lookup_sum_gt_engine_total" not in rec.invariant_violations:
                    rec.invariant_violations = list(rec.invariant_violations) + ["lookup_sum_gt_engine_total"]
                log.warning(
                    "request %s: lookup_sum=%d > engine_total=%d",
                    rec.request_id, lookup_sum, rec.engine_total_hit_tokens,
                )

        token_path = self.root / "tokens" / f"{rec.request_id}.parquet"
        prompt_tags = list(rec.span_tags) if rec.span_tags else [""] * len(rec.prompt_tokens)
        table = pa.table({
            "kind": ["prompt"] * len(rec.prompt_tokens) + ["response"] * len(rec.response_tokens),
            "token_id": list(rec.prompt_tokens) + list(rec.response_tokens),
            "tag": prompt_tags + [""] * len(rec.response_tokens),
        })
        pq.write_table(table, token_path)

        lookups_path = None
        if rec.span_lookups:
            lookups_path = self.root / "lookups" / f"{rec.request_id}.parquet"
            ltable = pa.table({
                "span_id": [l.span_id for l in rec.span_lookups],
                "chunk_hits": [l.chunk_hits for l in rec.span_lookups],
                "total_chunks": [l.total_chunks for l in rec.span_lookups],
                "hit_tokens": [l.hit_tokens for l in rec.span_lookups],
                "total_tokens": [l.total_tokens for l in rec.span_lookups],
            })
            pq.write_table(ltable, lookups_path)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO requests VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    rec.request_id, rec.session_id, rec.ts_start, rec.ts_end,
                    rec.prompt_token_count, rec.response_token_count,
                    rec.cache_read_tokens, rec.cache_recompute_tokens,
                    rec.engine_total_hit_tokens, rec.engine_load_tokens,
                    rec.tokens_recomputed_hkvd, rec.chunk_aligned_hit_tokens,
                    json.dumps(rec.invariant_violations),
                    rec.ttft_ms, rec.request_body_json, str(token_path),
                    str(lookups_path) if lookups_path else None,
                ),
            )

    def read_all(self) -> Iterator[RequestRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute("SELECT * FROM requests ORDER BY ts_start")
            for row in cur:
                # Reconstruct parquet paths from this store's root + the
                # canonical layout, NOT from the stored token_parquet_path
                # column. The stored column holds an absolute path captured
                # at write time — fragile across dir moves (the the harness
                # race left published swebench_verified captures with
                # absolute paths to a _raw/ staging dir that no longer
                # exists after the post-capture mv into the publication tree).
                token_path = self.root / "tokens" / f"{row['request_id']}.parquet"
                table = pq.read_table(token_path)
                kinds = table.column("kind").to_pylist()
                tokens = table.column("token_id").to_pylist()
                tags = table.column("tag").to_pylist()
                prompt = [t for k, t in zip(kinds, tokens) if k == "prompt"]
                response = [t for k, t in zip(kinds, tokens) if k == "response"]
                prompt_tags_raw = [g for k, g in zip(kinds, tags) if k == "prompt"]
                # Real tags are never the empty string ("" is the write-time placeholder
                # when span_tags was originally []). Restore that empty semantics on read
                # so a subsequent re-write doesn't trigger Invariant 1 spuriously.
                prompt_tags = prompt_tags_raw if any(prompt_tags_raw) else []
                lookups: list[ChunkHitMap] = []
                if row["lookups_parquet_path"]:
                    # Same root-relative reconstruction as token_path above.
                    lookups_path = self.root / "lookups" / f"{row['request_id']}.parquet"
                    ltable = pq.read_table(lookups_path)
                    for i in range(ltable.num_rows):
                        lookups.append(ChunkHitMap(
                            span_id=ltable.column("span_id")[i].as_py(),
                            chunk_hits=ltable.column("chunk_hits")[i].as_py(),
                            total_chunks=ltable.column("total_chunks")[i].as_py(),
                            hit_tokens=ltable.column("hit_tokens")[i].as_py(),
                            total_tokens=ltable.column("total_tokens")[i].as_py(),
                        ))
                yield RequestRecord(
                    request_id=row["request_id"], session_id=row["session_id"],
                    ts_start=row["ts_start"], ts_end=row["ts_end"],
                    prompt_tokens=prompt, response_tokens=response,
                    prompt_token_count=row["prompt_token_count"],
                    response_token_count=row["response_token_count"],
                    cache_read_tokens=row["cache_read_tokens"],
                    cache_recompute_tokens=row["cache_recompute_tokens"],
                    ttft_ms=row["ttft_ms"], request_body_json=row["request_body_json"],
                    span_tags=prompt_tags, span_lookups=lookups,
                    engine_total_hit_tokens=row["engine_total_hit_tokens"],
                    engine_load_tokens=row["engine_load_tokens"],
                    tokens_recomputed_hkvd=row["tokens_recomputed_hkvd"],
                    chunk_aligned_hit_tokens=row["chunk_aligned_hit_tokens"],
                    invariant_violations=json.loads(row["invariant_violations"] or "[]"),
                )
