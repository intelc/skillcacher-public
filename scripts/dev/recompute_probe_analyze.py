"""analyze the LMCACHE_BLEND_RECOMPUTE_RATIOS sweep.

Reads:
  benchmark/results/plan5_recompute_probe/r{label}/cacheblend.parquet
  benchmark/results/plan5_recompute_probe/r{label}/cacheblend/vllm.log
  benchmark/results/plan5_quality_spike_b_tp2/no_cache.parquet  (sample 0 baseline)

Writes:
  benchmark/results/plan5_recompute_probe/sweep_summary.csv
  paper/figures/recompute_ratio_tradeoff.pdf
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
PROBE_DIR = REPO / "benchmark/results/plan5_recompute_probe"
BASELINE_NC = REPO / "benchmark/results/plan5_quality_spike_b_tp2/no_cache.parquet"
FIG_DIR = REPO / "paper/figures"

RATIOS = [
    ("r015", 0.15),
    ("r05", 0.5),
    ("r10", 1.0),
]


def read_outputs(path: Path) -> dict[int, str]:
    """Return turn_index -> output text for sample_index=0."""
    t = pq.read_table(path)
    cols = {c: t[c].to_pylist() for c in
            ("turn_index", "sample_index", "text", "input_tokens", "output_tokens")}
    out = {}
    for i in range(t.num_rows):
        if cols["sample_index"][i] != 0:
            continue
        out[cols["turn_index"][i]] = {
            "text": cols["text"][i],
            "input_tokens": cols["input_tokens"][i],
            "output_tokens": cols["output_tokens"][i],
        }
    return out


def token_identity(a: str, b: str) -> float:
    """LCP over tokens, normalized by the longer length.

    Approximated at character level to avoid loading a tokenizer here —
    the per-turn comparison still discriminates on the same scale as the
    paper's identity_rate (which uses the model's tokenizer)."""
    if a == b:
        return 1.0
    n = min(len(a), len(b))
    lcp = 0
    for i in range(n):
        if a[i] != b[i]:
            break
        lcp += 1
    return lcp / max(len(a), len(b), 1)


VLLM_LOG_RE = re.compile(
    r"Total tokens (?P<total>\d+),.*?LMCache hit tokens: (?P<hit>\d+)"
)


def parse_rescue(vllm_log: Path) -> list[tuple[int, int]]:
    """Return list of (total, hit) tuples in request order."""
    pairs = []
    if not vllm_log.exists():
        return pairs
    with vllm_log.open() as f:
        for line in f:
            m = VLLM_LOG_RE.search(line)
            if m:
                pairs.append((int(m["total"]), int(m["hit"])))
    return pairs


def main() -> int:
    if not BASELINE_NC.exists():
        print(f"missing no_cache baseline: {BASELINE_NC}", file=sys.stderr)
        return 1
    nc = read_outputs(BASELINE_NC)
    print(f"no_cache baseline: {len(nc)} turns from {BASELINE_NC.name}")

    rows = []
    for label, ratio in RATIOS:
        cb_pq = PROBE_DIR / label / "cacheblend.parquet"
        vllm_log = PROBE_DIR / label / "cacheblend" / "vllm.log"
        if not cb_pq.exists():
            print(f"[r={ratio}] missing {cb_pq} — skipping", file=sys.stderr)
            continue
        cb = read_outputs(cb_pq)
        rescue = parse_rescue(vllm_log)
        for turn_idx, cb_row in sorted(cb.items()):
            nc_row = nc.get(turn_idx)
            if not nc_row:
                continue
            identity = token_identity(nc_row["text"] or "", cb_row["text"] or "")
            rows.append({
                "ratio": ratio,
                "label": label,
                "turn_index": turn_idx,
                "input_tokens": cb_row["input_tokens"],
                "output_tokens_nc": nc_row["output_tokens"],
                "output_tokens_cb": cb_row["output_tokens"],
                "identity": identity,
            })
        if rescue:
            print(f"[r={ratio}] vllm.log: {len(rescue)} LMCache lines, "
                  f"sum_total={sum(t for t,_ in rescue)}, "
                  f"sum_hit={sum(h for _,h in rescue)}")

    summary_csv = PROBE_DIR / "sweep_summary.csv"
    with summary_csv.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "ratio", "label", "turn_index", "input_tokens",
            "output_tokens_nc", "output_tokens_cb", "identity"
        ])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {summary_csv}")

    # Aggregate per ratio over substantive turns (input >= 256)
    by_ratio: dict[float, list[float]] = {}
    rescue_by_ratio: dict[float, tuple[int, int]] = {}
    for r in rows:
        if r["input_tokens"] < 256:
            continue
        by_ratio.setdefault(r["ratio"], []).append(r["identity"])

    for label, ratio in RATIOS:
        vllm_log = PROBE_DIR / label / "cacheblend" / "vllm.log"
        rescue = parse_rescue(vllm_log)
        # Only count requests with substantive prompts (>=256 tokens)
        subs = [(t, h) for (t, h) in rescue if t >= 256]
        if subs:
            total = sum(t for t, _ in subs)
            hit = sum(h for _, h in subs)
            rescue_by_ratio[ratio] = (total, hit)

    print("\n== summary ==")
    print(f"{'ratio':>6} {'n_turns':>8} {'mean_id':>9} {'rescue_total':>14} {'rescue_hit':>12} {'rescue_pct':>11}")
    for ratio in sorted(by_ratio):
        ids = by_ratio[ratio]
        mean_id = sum(ids) / len(ids)
        if ratio in rescue_by_ratio:
            tot, hit = rescue_by_ratio[ratio]
            pct = 100.0 * hit / tot if tot else 0.0
            print(f"{ratio:>6.2f} {len(ids):>8d} {mean_id:>9.4f} {tot:>14d} {hit:>12d} {pct:>10.1f}%")
        else:
            print(f"{ratio:>6.2f} {len(ids):>8d} {mean_id:>9.4f} {'-':>14} {'-':>12} {'-':>11}")

    # Plot
    if by_ratio and rescue_by_ratio:
        ratios_sorted = sorted(by_ratio)
        mean_ids = [sum(by_ratio[r]) / len(by_ratio[r]) for r in ratios_sorted]
        rescue_pcts = [
            100.0 * rescue_by_ratio[r][1] / rescue_by_ratio[r][0]
            if r in rescue_by_ratio and rescue_by_ratio[r][0] else 0.0
            for r in ratios_sorted
        ]

        matplotlib.rcParams.update({
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        })
        fig, ax_id = plt.subplots(figsize=(3.3, 2.3))
        ax_rs = ax_id.twinx()

        ln_id = ax_id.plot(ratios_sorted, mean_ids, marker="o",
                            color="#2c7fb8", linewidth=1.3,
                            label="mean identity")
        ln_rs = ax_rs.plot(ratios_sorted, rescue_pcts, marker="s",
                            color="#d62728", linewidth=1.3,
                            linestyle="--", label="rescue %")

        ax_id.set_xlabel("LMCACHE_BLEND_RECOMPUTE_RATIOS")
        ax_id.set_ylabel("mean token-identity rate", color="#2c7fb8")
        ax_id.tick_params(axis="y", labelcolor="#2c7fb8")
        ax_id.set_ylim(-0.02, 1.05)
        ax_rs.set_ylabel("rescue % (substantive)", color="#d62728")
        ax_rs.tick_params(axis="y", labelcolor="#d62728")
        ax_rs.set_ylim(-2, 105)

        ax_id.grid(alpha=0.2, linewidth=0.5)
        ax_id.set_axisbelow(True)
        ax_id.set_xticks(ratios_sorted)

        lines = ln_id + ln_rs
        labels = [l.get_label() for l in lines]
        ax_id.legend(lines, labels, loc="center right", frameon=False,
                     fontsize=7)

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        out_pdf = FIG_DIR / "recompute_ratio_tradeoff.pdf"
        fig.savefig(out_pdf)
        plt.close(fig)
        print(f"\nwrote figure: {out_pdf}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
