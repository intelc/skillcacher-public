"""pre-publish audit for the ClaudeCodeTrace dataset.

Walks a fixture root and surfaces any residual operational artifacts —
proxy URLs, API keys, Tailscale identifiers, RunPod IDs, HF tokens —
that survived `redact.py`. Catches three classes of leak:

1. **Text files** (.json, .log, .txt, .md) — re-grep with the patterns
   from `scripts.redact`, fail on any match.
2. **Parquet files** — load every string-typed column, grep each value.
   Trace-store token parquets are int-typed (`token_id`) so they're
   passed through unchanged; the audit only fires on text columns.
3. **traces.sqlite** — extract `request_body_json` (and any other TEXT
   column) from the `requests` table, grep each row.

Three modes:

- ``--dry-run`` (default): scan only, write a per-fixture report.
- ``--apply``: re-run `redact.redact_text` in place over every text file,
  then re-scan. Useful when an old capture predates a redaction-pattern
  addition (e.g., the capture tailscale-auth-key pattern).
- ``--strict``: exit non-zero when residual violations are found after
  the chosen mode. Used by CI / the publication pipeline.

Usage:
    python -m scripts.publish_claudecode_trace tests/fixtures/claude_code_real \\
        --report benchmark/results/audit/plan4_publish_audit.md
    python -m scripts.publish_claudecode_trace tests/fixtures/claude_code_real \\
        --apply --strict
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from scripts.redact import (
    API_KEY,
    BEARER,
    CC_VERSION,
    HF_TOKEN,
    RUNPOD_KEY,
    RUNPOD_URL,
    SYSTEM_HASH,
    TAILSCALE_AUTH_KEY,
    TAILSCALE_HOST,
    redact_text,
)

# Pattern table for the audit. Keep keys stable — the test surface
# matches on these names. RUNPOD_POD_ID is intentionally excluded:
# `redact.py` doesn't emit replacements for it (the pattern is too
# permissive to use globally), and including it here would noisy-match
# the SWE-Bench instance IDs we *do* want preserved.
AUDIT_PATTERNS: dict[str, re.Pattern] = {
    "runpod_url": RUNPOD_URL,
    "tailscale_host": TAILSCALE_HOST,
    "tailscale_auth_key": TAILSCALE_AUTH_KEY,
    "hf_token": HF_TOKEN,
    "runpod_key": RUNPOD_KEY,
    "api_key": API_KEY,
    "bearer": BEARER,
    "cc_version": CC_VERSION,
    "system_hash": SYSTEM_HASH,
}

# File-extension surface for the text-redaction pass. Mirrors what
# `redact.py main()` walks. Parquet + sqlite are inspected separately.
TEXT_EXTS = (".json", ".log", ".txt", ".md")


@dataclass
class Violation:
    path: str
    location: str  # "text", "sqlite:<table>.<col>:row<N>", "parquet:<col>:row<N>"
    pattern: str
    snippet: str  # first 80 chars of the offending match for the report

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "location": self.location,
            "pattern": self.pattern,
            "snippet": self.snippet,
        }


@dataclass
class AuditReport:
    root: str
    text_files_scanned: int = 0
    sqlite_files_scanned: int = 0
    parquet_files_scanned: int = 0
    files_redacted: int = 0
    violations: list[Violation] = field(default_factory=list)
    by_pattern: Counter = field(default_factory=Counter)

    def add(self, v: Violation) -> None:
        self.violations.append(v)
        self.by_pattern[v.pattern] += 1

    @property
    def clean(self) -> bool:
        return not self.violations


def scan_text(text: str, *, path: str, location: str = "text") -> list[Violation]:
    """Scan a single text blob and return any leaks. Skips matches whose
    captured value is a redaction marker (`<REDACTED_…>`): a few of the
    redact.py patterns (cc_version, system_hash) match `"key": "<value>"`
    JSON shape and the replacement text retains the same shape, so the
    pattern matches its own marker. That's idempotent for redact.py
    (running it again leaves the marker unchanged) but a false positive
    for the audit. Filter those out here so the audit only fires on
    REAL residual leaks."""
    out: list[Violation] = []
    for name, pat in AUDIT_PATTERNS.items():
        for m in pat.finditer(text):
            snip = m.group(0)[:80]
            if "<REDACTED" in snip:
                continue
            out.append(Violation(path=path, location=location,
                                 pattern=name, snippet=snip))
    return out


def scan_sqlite(path: Path) -> list[Violation]:
    """Walk every TEXT column of every table in the sqlite file and
    audit each row's value. Most relevant column for the trace store is
    `requests.request_body_json`, but we generalize so future schema
    additions don't silently slip leaks past."""
    out: list[Violation] = []
    try:
        conn = sqlite3.connect(path)
    except sqlite3.OperationalError as e:
        out.append(Violation(path=str(path), location="sqlite:open",
                             pattern="open_error", snippet=str(e)[:80]))
        return out
    try:
        conn.row_factory = sqlite3.Row
        tables = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()]
        for table in tables:
            cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
            text_cols = [c[1] for c in cols if (c[2] or "").upper() in ("TEXT", "")]
            if not text_cols:
                continue
            qcols = ", ".join(f'"{c}"' for c in text_cols)
            for i, row in enumerate(conn.execute(f"SELECT {qcols} FROM {table}")):
                for col_name, val in zip(text_cols, row):
                    if not isinstance(val, str):
                        continue
                    out.extend(
                        scan_text(val, path=str(path),
                                  location=f"sqlite:{table}.{col_name}:row{i}")
                    )
    finally:
        conn.close()
    return out


def scan_parquet(path: Path) -> list[Violation]:
    """Walk every string-typed column in the parquet file."""
    import pyarrow.parquet as pq
    try:
        table = pq.read_table(path)
    except Exception as e:
        return [Violation(path=str(path), location="parquet:open",
                          pattern="open_error", snippet=str(e)[:80])]
    out: list[Violation] = []
    for col_name in table.column_names:
        col = table.column(col_name)
        # Only string-ish columns. Token-id columns are int64; skip.
        if not str(col.type).startswith("string") and not str(col.type).startswith("large_string"):
            continue
        for i, val in enumerate(col.to_pylist()):
            if not isinstance(val, str):
                continue
            out.extend(
                scan_text(val, path=str(path),
                          location=f"parquet:{col_name}:row{i}")
            )
    return out


def audit_root(root: Path, *, apply: bool) -> AuditReport:
    """Scan every text file + sqlite + parquet under `root`. With
    ``apply=True``, runs `redact_text` in place over text files BEFORE
    scanning, so the audit reflects the post-redaction state."""
    report = AuditReport(root=str(root))

    # 1. Text files — optionally re-redact, then scan.
    for ext in TEXT_EXTS:
        for path in sorted(root.rglob(f"*{ext}")):
            try:
                raw = path.read_text()
            except (UnicodeDecodeError, OSError):
                continue
            if apply:
                new = redact_text(raw)
                if new != raw:
                    path.write_text(new)
                    report.files_redacted += 1
                raw = new
            report.text_files_scanned += 1
            for v in scan_text(raw, path=str(path)):
                report.add(v)

    # 2. SQLite — scan only (writing back into a sqlite blob has its own
    # risks; if redaction is needed, regenerate from the source instead).
    for path in sorted(root.rglob("*.sqlite")):
        report.sqlite_files_scanned += 1
        for v in scan_sqlite(path):
            report.add(v)

    # 3. Parquet — same treatment as sqlite (scan only).
    for path in sorted(root.rglob("*.parquet")):
        report.parquet_files_scanned += 1
        for v in scan_parquet(path):
            report.add(v)

    return report


def write_report(report: AuditReport, out: Path) -> None:
    """Render the audit report as Markdown."""
    out.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# ClaudeCodeTrace pre-publish audit")
    lines.append("")
    lines.append(f"**Root:** `{report.root}`")
    lines.append(f"**Text files scanned:** {report.text_files_scanned}")
    lines.append(f"**SQLite files scanned:** {report.sqlite_files_scanned}")
    lines.append(f"**Parquet files scanned:** {report.parquet_files_scanned}")
    lines.append(f"**Files redacted (--apply):** {report.files_redacted}")
    lines.append(f"**Total violations:** {len(report.violations)}")
    lines.append("")
    if report.clean:
        lines.append("**Status: CLEAN — no residual artifacts found.**")
    else:
        lines.append("**Status: DIRTY — residual artifacts found, see below.**")
    lines.append("")
    if report.by_pattern:
        lines.append("## Violations by pattern")
        lines.append("")
        lines.append("| Pattern | Count |")
        lines.append("|---|---:|")
        for pat, cnt in sorted(report.by_pattern.items(), key=lambda kv: -kv[1]):
            lines.append(f"| `{pat}` | {cnt} |")
        lines.append("")
    if report.violations:
        # Group by file for readability; cap at 200 entries to keep the
        # report small (a fully-broken capture would otherwise produce a
        # 100K-line markdown file no one will read).
        by_file: dict[str, list[Violation]] = defaultdict(list)
        for v in report.violations:
            by_file[v.path].append(v)
        lines.append("## Violations by file (first 200)")
        lines.append("")
        shown = 0
        for path, vs in sorted(by_file.items()):
            if shown >= 200:
                lines.append(f"... ({len(report.violations) - shown} more violations omitted)")
                break
            lines.append(f"### `{path}`")
            for v in vs[:20]:
                if shown >= 200:
                    break
                lines.append(f"- `{v.pattern}` at `{v.location}` — `{v.snippet}`")
                shown += 1
            if len(vs) > 20:
                lines.append(f"- ... ({len(vs) - 20} more in this file)")
    out.write_text("\n".join(lines) + "\n")


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("root", type=Path, help="Fixture root to audit.")
    p.add_argument("--apply", action="store_true",
                   help="Re-run redact_text on text files in place before scanning.")
    p.add_argument("--strict", action="store_true",
                   help="Exit 1 if any violations remain after the chosen mode.")
    p.add_argument("--report", type=Path, default=None,
                   help="Markdown report path (default: stdout summary only).")
    p.add_argument("--json", type=Path, default=None,
                   help="Optional JSON dump of all violations.")
    args = p.parse_args(argv[1:])

    if not args.root.exists():
        print(f"[publish-audit] root {args.root} does not exist", file=sys.stderr)
        return 2

    report = audit_root(args.root, apply=args.apply)
    print(
        f"[publish-audit] {report.text_files_scanned} text + "
        f"{report.sqlite_files_scanned} sqlite + "
        f"{report.parquet_files_scanned} parquet files scanned; "
        f"{len(report.violations)} violations "
        f"({'CLEAN' if report.clean else 'DIRTY'})",
        file=sys.stderr,
    )
    if args.report:
        write_report(report, args.report)
        print(f"[publish-audit] report → {args.report}", file=sys.stderr)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(
            {
                "root": report.root,
                "text_files_scanned": report.text_files_scanned,
                "sqlite_files_scanned": report.sqlite_files_scanned,
                "parquet_files_scanned": report.parquet_files_scanned,
                "files_redacted": report.files_redacted,
                "by_pattern": dict(report.by_pattern),
                "violations": [v.as_dict() for v in report.violations],
            },
            indent=2,
        ))
    if args.strict and not report.clean:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
