"""round-trip test for the pre-publish audit script.

Constructs a synthetic fixture dir containing every artifact class the
audit is supposed to catch (proxy URLs, API keys, Bearer headers, CC
versions, system-prompt hashes, Tailscale auth-keys + hostnames, RunPod
keys, HF tokens), spread across .json / .log / .md text files plus a
synthetic traces.sqlite and a string-typed parquet.

Asserts:

1. The dry-run scan finds AT LEAST one violation per artifact class —
   if a future refactor accidentally drops a pattern from
   `AUDIT_PATTERNS`, this test fails.
2. After ``--apply``, every text-file violation is gone (redact_text
   replaced them).
3. SQLite + parquet violations are still present after ``--apply``
   (audit doesn't rewrite binary stores; the test confirms that
   contract — a future regression that silently rewrites them would
   risk corrupting the dataset).
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.publish_claudecode_trace import audit_root, AUDIT_PATTERNS


# Concrete leaks for each pattern. Each string MUST trigger exactly the
# named pattern in `redact.py` / `AUDIT_PATTERNS`.
LEAKS: dict[str, str] = {
    "runpod_url": "https://abcdefgh-8000.proxy.runpod.net/v1/messages",
    "tailscale_host": "skillcacher-bench-cacheblend.tailnet-abc123.ts.net",
    "tailscale_auth_key": "tskey-auth-kABCDEFGHIJKLMNOPQRSTUVWXYZ012345678",
    "hf_token": "hf_abcdefghijklmnopqrstuvwxyz0123456789ABCD",
    "runpod_key": "RPA_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ABCDEFGHIJ",
    "api_key": "sk-ant-api03-AAAAAAAAAAAAAAAAAAAAAAAAA",
    "bearer": '"Authorization": "Bearer abcdefghij-_TOKEN"',
    "cc_version": '"cli_version": "2.1.0.abcdefg"',
    "system_hash": '"system_prompt_hash": "deadbeefcafef00d"',
}


@pytest.fixture
def fixture_root(tmp_path: Path) -> Path:
    """Build a synthetic fixture dir laid out like a real capture."""
    root = tmp_path / "claude_code_real"
    capture = root / "swebench_verified" / "synthetic-task-001"
    capture.mkdir(parents=True)

    # 1. A JSON file with a leaky proxy URL + bearer header + cc_version.
    (capture / "_claude_stdout.json").write_text(json.dumps({
        "transport": LEAKS["runpod_url"],
        "headers": {"Authorization": "Bearer abcdefghij-_TOKEN"},
        "cli_version": "2.1.0.abcdefg",
    }))

    # 2. A boot log dumping the tailscale auth-key + hostname.
    (capture / "oneshot_boot.log").write_text(
        "+ tailscale up --auth-key " + LEAKS["tailscale_auth_key"] + "\n"
        "+ resolved " + LEAKS["tailscale_host"] + "\n"
        "+ exporting RUNPOD_API_KEY=" + LEAKS["runpod_key"] + "\n"
    )

    # 3. A markdown notes file with an HF token + sk-ant key.
    (capture / "notes.md").write_text(
        "Used HF_TOKEN=" + LEAKS["hf_token"] + " for the pre-pull.\n"
        "Anthropic key: " + LEAKS["api_key"] + "\n"
    )

    # 4. A txt file with the system prompt hash.
    (capture / "manifest.txt").write_text(
        "{ " + LEAKS["system_hash"] + " }\n"
    )

    # 5. A parquet with a string column containing a leaky body.
    parquet_path = capture / "_traces" / "tokens" / "req_synthetic.parquet"
    parquet_path.parent.mkdir(parents=True)
    pq.write_table(
        pa.table({
            "request_id": ["req_synth_001"],
            "body_text": ["body referencing " + LEAKS["runpod_url"]],
        }),
        parquet_path,
    )

    # 6. A traces.sqlite with the standard requests schema and one row
    #    that has a leaky `request_body_json`.
    db_path = capture / "_traces" / "traces.sqlite"
    with sqlite3.connect(db_path) as conn:
        conn.executescript("""
        CREATE TABLE requests (
          request_id TEXT PRIMARY KEY,
          request_body_json TEXT,
          ts_start REAL
        );
        """)
        conn.execute(
            "INSERT INTO requests VALUES (?, ?, ?)",
            ("req_synth_001",
             json.dumps({"transport": LEAKS["runpod_url"],
                         "system_prompt_hash": "deadbeefcafef00d"}),
             0.0),
        )

    return root


def test_audit_catches_each_artifact_class(fixture_root: Path):
    report = audit_root(fixture_root, apply=False)
    seen_patterns = set(report.by_pattern.keys())
    # Every pattern with a synthetic leak above must appear at least once.
    for name in LEAKS:
        assert name in seen_patterns, (
            f"audit missed pattern {name!r}; saw only {seen_patterns}"
        )
    # The text-file scan + sqlite + parquet should each produce hits.
    assert report.text_files_scanned == 4  # json + log + md + txt
    assert report.sqlite_files_scanned == 1
    assert report.parquet_files_scanned == 1
    assert any(v.location.startswith("sqlite:") for v in report.violations)
    assert any(v.location.startswith("parquet:") for v in report.violations)
    assert not report.clean


def test_apply_clears_text_file_violations(fixture_root: Path):
    report = audit_root(fixture_root, apply=True)
    # All the text-file artifact classes have replacements in `redact_text`,
    # so post-apply there should be ZERO text-location violations.
    text_only = [v for v in report.violations if v.location == "text"]
    assert text_only == [], (
        f"--apply left {len(text_only)} text-file violations; "
        f"first few: {[v.as_dict() for v in text_only[:3]]}"
    )
    # files_redacted counts how many text files were rewritten by
    # redact_text — at least 4 (one per text file we wrote with leaks).
    assert report.files_redacted >= 4


def test_apply_does_not_rewrite_binary_stores(fixture_root: Path):
    """Critical contract: --apply must NOT touch sqlite or parquet
    files. Rewriting them naively would corrupt the column types or
    silently re-encode rows in ways that break downstream consumers.
    Verify that sqlite/parquet violations are still present after apply."""
    audit_root(fixture_root, apply=True)
    re_scan = audit_root(fixture_root, apply=False)
    sqlite_violations = [v for v in re_scan.violations if v.location.startswith("sqlite:")]
    parquet_violations = [v for v in re_scan.violations if v.location.startswith("parquet:")]
    assert sqlite_violations, (
        "expected sqlite violations to survive --apply (binary stores not rewritten)"
    )
    assert parquet_violations, (
        "expected parquet violations to survive --apply (binary stores not rewritten)"
    )


def test_audit_clean_on_empty_fixture(tmp_path: Path):
    empty = tmp_path / "empty_fixture"
    empty.mkdir()
    report = audit_root(empty, apply=False)
    assert report.clean
    assert report.text_files_scanned == 0
    assert report.sqlite_files_scanned == 0
    assert report.parquet_files_scanned == 0


def test_pattern_table_covers_redact_replacements():
    """Drift guard: every artifact class that `redact.py` knows how to
    rewrite should have a matching audit pattern, otherwise we'd silently
    miss a leak that redact_text *thinks* it caught.

    Today this is one-to-one — verify the pattern keys match the names
    that audit_root reports on, so a future addition to `redact.py`
    forces the test author to add it here too."""
    expected = {
        "runpod_url", "tailscale_host", "tailscale_auth_key",
        "hf_token", "runpod_key", "api_key", "bearer",
        "cc_version", "system_hash",
    }
    assert set(AUDIT_PATTERNS) == expected, (
        f"audit pattern set drifted: have {set(AUDIT_PATTERNS)}, expected {expected}"
    )
