"""SQLite-backed registry of mined spans + their cached KV blob ids.

Bridges the structural prefix index (byte-level) and Controller.Lookup /
Controller.Pin (token-level)."""
from __future__ import annotations

import sqlite3
import struct
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Span:
    span_id: str
    skill_id: str
    anchor_bytes: int
    token_ids: list[int]
    registered_at: float
    last_lookup_at: float | None
    lookup_hit_count: int
    lookup_total_count: int
    # marks how this span was discovered. "structural" is the
    # original SkillPrefixIndex byte-prefix path; "statistical" is the
    # statistical_miner suffix-array path. Lookup never filters on source
    # — both flavours participate equally — but having the source on the
    # row lets us count miner contribution in the §3 ablation report.
    source: str = "structural"


def _pack_tokens(ids: list[int]) -> bytes:
    return struct.pack(f">{len(ids)}I", *ids)


def _unpack_tokens(blob: bytes) -> list[int]:
    n = len(blob) // 4
    return list(struct.unpack(f">{n}I", blob))


class SpanRegistry:
    SCHEMA = """
    CREATE TABLE IF NOT EXISTS spans (
        span_id TEXT PRIMARY KEY,
        skill_id TEXT NOT NULL,
        anchor_bytes INTEGER NOT NULL,
        token_ids BLOB NOT NULL,
        registered_at REAL NOT NULL,
        last_lookup_at REAL,
        lookup_hit_count INTEGER NOT NULL DEFAULT 0,
        lookup_total_count INTEGER NOT NULL DEFAULT 0,
        source TEXT NOT NULL DEFAULT 'structural'
    );
    CREATE INDEX IF NOT EXISTS idx_skill ON spans(skill_id);
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def init_schema(self) -> None:
        """Create tables if missing, then run any column-level migrations
        needed to bring an older DB up to the current shape. the harness
        added the `source` column; older registries created before then
        get it added here. The source-index is created AFTER the
        migration so it doesn't fail on legacy DBs."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)
            cur = conn.execute("PRAGMA table_info(spans)")
            cols = {r[1] for r in cur.fetchall()}
            if "source" not in cols:
                conn.execute(
                    "ALTER TABLE spans ADD COLUMN source TEXT NOT NULL DEFAULT 'structural'"
                )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_source ON spans(source)")

    def register(
        self, span_id: str, *,
        token_ids: list[int], skill_id: str, anchor_bytes: int,
        source: str = "structural",
    ) -> None:
        """Insert (or upsert) a span. ``source`` defaults to "structural"
        (the original skill-prefix path); pass "statistical" for spans
        from the §3 miner. Lookup is source-agnostic; this is purely
        provenance metadata that the ablation report buckets on."""
        if source not in ("structural", "statistical"):
            raise ValueError(
                f"source must be 'structural' or 'statistical', got {source!r}"
            )
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO spans (span_id, skill_id, anchor_bytes, token_ids,
                                       registered_at, source)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(span_id) DO UPDATE SET
                     token_ids=excluded.token_ids,
                     skill_id=excluded.skill_id,
                     anchor_bytes=excluded.anchor_bytes,
                     registered_at=excluded.registered_at,
                     source=excluded.source""",
                (span_id, skill_id, anchor_bytes, _pack_tokens(token_ids), now, source),
            )

    def get(self, span_id: str) -> Span | None:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM spans WHERE span_id = ?", (span_id,)).fetchone()
        if not row:
            return None
        return Span(
            span_id=row["span_id"], skill_id=row["skill_id"],
            anchor_bytes=row["anchor_bytes"], token_ids=_unpack_tokens(row["token_ids"]),
            registered_at=row["registered_at"], last_lookup_at=row["last_lookup_at"],
            lookup_hit_count=row["lookup_hit_count"], lookup_total_count=row["lookup_total_count"],
            source=row["source"] if "source" in row.keys() else "structural",
        )

    def spans_referenced_by(self, *, skill_ids: list[str]) -> list[Span]:
        if not skill_ids:
            return []
        qs = ",".join("?" * len(skill_ids))
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(f"SELECT * FROM spans WHERE skill_id IN ({qs})", skill_ids).fetchall()
        return [
            Span(
                span_id=r["span_id"], skill_id=r["skill_id"],
                anchor_bytes=r["anchor_bytes"], token_ids=_unpack_tokens(r["token_ids"]),
                registered_at=r["registered_at"], last_lookup_at=r["last_lookup_at"],
                lookup_hit_count=r["lookup_hit_count"], lookup_total_count=r["lookup_total_count"],
                source=r["source"] if "source" in r.keys() else "structural",
            )
            for r in rows
        ]

    def all_spans(self, *, source: str | None = None) -> list[Span]:
        """Fetch every span; optionally filter by source. Used by the
        the lookup-time wiring to enumerate statistical entries
        (which don't have a meaningful skill_id to scope on) and by the
        ablation report to bucket counts by source."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            if source is None:
                rows = conn.execute("SELECT * FROM spans").fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM spans WHERE source = ?", (source,)
                ).fetchall()
        return [
            Span(
                span_id=r["span_id"], skill_id=r["skill_id"],
                anchor_bytes=r["anchor_bytes"], token_ids=_unpack_tokens(r["token_ids"]),
                registered_at=r["registered_at"], last_lookup_at=r["last_lookup_at"],
                lookup_hit_count=r["lookup_hit_count"], lookup_total_count=r["lookup_total_count"],
                source=r["source"] if "source" in r.keys() else "structural",
            )
            for r in rows
        ]

    def record_lookup_result(self, span_id: str, *, hit_tokens: int, total_tokens: int) -> None:
        now = time.time()
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """UPDATE spans SET
                     last_lookup_at = ?,
                     lookup_hit_count = lookup_hit_count + ?,
                     lookup_total_count = lookup_total_count + 1
                   WHERE span_id = ?""",
                (now, hit_tokens, span_id),
            )

    def prune(self, *, older_than: float) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM spans WHERE last_lookup_at IS NOT NULL AND last_lookup_at < ?",
                (older_than,),
            )
            return cur.rowcount
