"""Static CSV + HTML report writer. No JS; inline SVG bars."""
from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path

from skillcacher.bench.metrics import AggregateRates


def write_report(
    out_dir: Path, *, workload: str, fixture_set: str,
    per_condition: dict[str, AggregateRates],
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    csv_path = out_dir / "summary.csv"
    fieldnames = ["condition"] + list(asdict(next(iter(per_condition.values()))).keys())
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cond_name, agg in per_condition.items():
            row = {"condition": cond_name, **asdict(agg)}
            w.writerow(row)

    # HTML
    html = _render_html(workload, fixture_set, per_condition)
    (out_dir / "report.html").write_text(html)


def _render_html(workload: str, fixture_set: str, per_condition: dict[str, AggregateRates]) -> str:
    bars = ""
    for cond_name, agg in per_condition.items():
        bars += _render_bar_group(cond_name, agg)
    table = _render_table(per_condition)
    return f"""<!DOCTYPE html>
<html>
<head>
<title>skillcacher bench: {workload} ({fixture_set})</title>
<style>
body {{ font-family: -apple-system, sans-serif; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
h2 {{ margin-top: 2em; }}
.bar-row {{ display: flex; align-items: center; margin: 0.4em 0; }}
.bar-label {{ width: 14em; font-size: 0.9em; }}
.bar-track {{ flex: 1; background: #eee; height: 1.4em; position: relative; }}
.bar-fill {{ background: #2c7be5; height: 100%; }}
.bar-value {{ margin-left: 0.5em; font-variant-numeric: tabular-nums; font-size: 0.85em; }}
table {{ border-collapse: collapse; width: 100%; margin-top: 1em; }}
th, td {{ text-align: left; padding: 0.4em 0.8em; border-bottom: 1px solid #ddd; font-size: 0.9em; }}
th {{ background: #f7f7f7; }}
.condition-block {{ border: 1px solid #ddd; padding: 1em; margin: 1em 0; }}
</style>
</head>
<body>
<h1>skillcacher bench: {workload}</h1>
<p>Fixture set: <code>{fixture_set}</code></p>
<h2>E1 — Decomposed hit rate per condition</h2>
{bars}
<h2>Aggregate table</h2>
{table}
</body>
</html>
"""


def _render_bar_group(condition: str, agg: AggregateRates) -> str:
    rows = ""
    for label, val in [
        ("skill_hit_rate", agg.skill_hit_rate),
        ("non_skill_hit_rate", agg.non_skill_hit_rate),
        ("overall_hit_rate", agg.overall_hit_rate),
    ]:
        pct = max(0.0, min(1.0, val)) * 100
        rows += (
            f'<div class="bar-row">'
            f'<div class="bar-label">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill" style="width: {pct:.1f}%"></div></div>'
            f'<div class="bar-value">{val:.3f}</div>'
            f'</div>'
        )
    return f'<div class="condition-block"><h3>{condition} (n={agg.n_requests})</h3>{rows}</div>'


def _render_table(per_condition: dict[str, AggregateRates]) -> str:
    rows = ""
    for cond_name, agg in per_condition.items():
        rows += (
            f"<tr><td>{cond_name}</td>"
            f"<td>{agg.skill_hit_rate:.3f}</td>"
            f"<td>{agg.non_skill_hit_rate:.3f}</td>"
            f"<td>{agg.overall_hit_rate:.3f}</td>"
            f"<td>{agg.skill_hit_tokens:,}</td>"
            f"<td>{agg.non_skill_hit_tokens:,}</td>"
            f"<td>{agg.prompt_token_count:,}</td>"
            f"<td>{agg.n_requests}</td></tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>condition</th><th>skill_hit_rate</th><th>non_skill_hit_rate</th><th>overall_hit_rate</th>"
        "<th>skill_hit_tokens</th><th>non_skill_hit_tokens</th><th>prompt_tokens</th><th>n</th>"
        "</tr></thead><tbody>" + rows + "</tbody></table>"
    )
