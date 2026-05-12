"""Tests for the CSV + HTML report writer."""
from pathlib import Path

from skillcacher.bench.report import write_report
from skillcacher.bench.metrics import AggregateRates


def _agg(skill=0.5, non_skill=0.5, overall=0.5, n=10):
    return AggregateRates(
        skill_hit_rate=skill, non_skill_hit_rate=non_skill, overall_hit_rate=overall,
        skill_hit_tokens=int(skill * 1000), non_skill_hit_tokens=int(non_skill * 1000),
        total_hit_tokens=int(overall * 2000),
        skill_total_tokens=1000, non_skill_total_tokens=1000, prompt_token_count=2000,
        n_requests=n,
    )


def test_write_report_emits_csv_and_html(tmp_path):
    per_condition = {
        "no_cache": _agg(0.0, 0.0, 0.0),
        "prefix_cache": _agg(0.2, 0.8, 0.5),
        "cacheblend": _agg(0.85, 0.85, 0.85),
    }
    write_report(tmp_path, workload="swebench_lite_20", fixture_set="swebench_lite", per_condition=per_condition)
    assert (tmp_path / "report.html").exists()
    assert (tmp_path / "summary.csv").exists()
    csv_text = (tmp_path / "summary.csv").read_text()
    assert "skill_hit_rate" in csv_text
    assert "cacheblend" in csv_text


def test_html_contains_per_condition_bars(tmp_path):
    per_condition = {
        "no_cache": _agg(0.0, 0.0, 0.0),
        "cacheblend": _agg(0.85, 0.85, 0.85),
    }
    write_report(tmp_path, workload="w", fixture_set="fs", per_condition=per_condition)
    html = (tmp_path / "report.html").read_text()
    assert "no_cache" in html and "cacheblend" in html
    assert "skill_hit_rate" in html
