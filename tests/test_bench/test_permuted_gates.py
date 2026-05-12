"""Tests for the permuted-workload pass-criterion gate checker."""
from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

from skillcacher.bench.permuted_gates import check_gate, GateResult


# Header order matches `bench/report.py:write_report`'s CSV.
CSV_FIELDS = [
    "condition", "skill_hit_rate", "non_skill_hit_rate", "overall_hit_rate",
    "skill_hit_tokens", "non_skill_hit_tokens", "total_hit_tokens",
    "skill_total_tokens", "non_skill_total_tokens", "prompt_token_count",
    "n_requests",
]


def _write_summary(path: Path, rows: dict[str, dict]) -> None:
    """Write a tiny summary.csv. `rows` maps condition → {col: value}; only
    overall_hit_rate is required, the rest get sensible zeros."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for cond, vals in rows.items():
            row = {k: 0 for k in CSV_FIELDS}
            row["condition"] = cond
            row.update(vals)
            w.writerow(row)


def test_full_5_pass(tmp_path):
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "no_cache": {"overall_hit_rate": 0.0},
        "prefix_cache": {"overall_hit_rate": 0.10},
        "cacheblend": {"overall_hit_rate": 0.95},
    })
    result = check_gate(csv_path, workload="full_5")
    assert result.verdict == "PASS"
    assert result.exit_code == 0
    assert "PASS" in result.markdown


def test_full_5_fail_cacheblend_low(tmp_path):
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "prefix_cache": {"overall_hit_rate": 0.10},
        "cacheblend": {"overall_hit_rate": 0.70},
    })
    result = check_gate(csv_path, workload="full_5")
    assert result.verdict == "FAIL"
    assert result.exit_code == 1
    assert "cacheblend" in result.markdown.lower()


def test_full_5_fail_prefix_high(tmp_path):
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "prefix_cache": {"overall_hit_rate": 0.30},
        "cacheblend": {"overall_hit_rate": 0.95},
    })
    result = check_gate(csv_path, workload="full_5")
    assert result.verdict == "FAIL"
    assert "prefix_cache" in result.markdown.lower()


def test_full_5_borderline_marked(tmp_path):
    """cacheblend at 0.79 (1pp under threshold of 0.80) → FAIL with BORDERLINE."""
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "prefix_cache": {"overall_hit_rate": 0.10},
        "cacheblend": {"overall_hit_rate": 0.79},
    })
    result = check_gate(csv_path, workload="full_5")
    assert result.verdict == "FAIL"
    assert "BORDERLINE" in result.markdown


def test_partial_20_pass(tmp_path):
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "prefix_cache": {"overall_hit_rate": 0.32},
        "cacheblend": {"overall_hit_rate": 0.70},
    })
    result = check_gate(csv_path, workload="partial_20")
    assert result.verdict == "PASS"
    assert "delta" in result.markdown.lower()


def test_partial_20_fail_delta_low(tmp_path):
    """delta = 0.62 - 0.34 = 28pp; under 30pp threshold."""
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "prefix_cache": {"overall_hit_rate": 0.34},
        "cacheblend": {"overall_hit_rate": 0.62},
    })
    result = check_gate(csv_path, workload="partial_20")
    assert result.verdict == "FAIL"
    assert "BORDERLINE" in result.markdown  # within 2pp of threshold


def test_partial_20_fail_cacheblend_too_low(tmp_path):
    """delta is 35pp (passes) but cacheblend 0.55 < 0.60 absolute floor."""
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "prefix_cache": {"overall_hit_rate": 0.20},
        "cacheblend": {"overall_hit_rate": 0.55},
    })
    result = check_gate(csv_path, workload="partial_20")
    assert result.verdict == "FAIL"


def test_summary_csv_missing(tmp_path):
    result = check_gate(tmp_path / "no_such.csv", workload="full_5")
    assert result.verdict == "GATE_ERROR"
    assert result.exit_code == 2


def test_condition_row_missing(tmp_path):
    """summary.csv has cacheblend but no prefix_cache → FAIL with reason."""
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "cacheblend": {"overall_hit_rate": 0.95},
    })
    result = check_gate(csv_path, workload="full_5")
    assert result.verdict == "FAIL"
    assert "prefix_cache" in result.markdown.lower()


def test_cli_main_exit_code(tmp_path):
    """python -m skillcacher.bench.permuted_gates <csv> --workload full_5
    exits 0/1/2 matching the verdict."""
    csv_path = tmp_path / "summary.csv"
    _write_summary(csv_path, {
        "prefix_cache": {"overall_hit_rate": 0.10},
        "cacheblend": {"overall_hit_rate": 0.95},
    })
    r = subprocess.run(
        [sys.executable, "-m", "skillcacher.bench.permuted_gates",
         str(csv_path), "--workload", "full_5"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0
    assert "PASS" in r.stdout
