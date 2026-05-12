"""Generate paper figures from benchmark/ data.

Outputs PDFs into paper/figures/ that main.tex includes.

Run via:
    make figures
or:
    .venv/bin/python paper/scripts/make_figures.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

_PAPER_DIR = Path(__file__).resolve().parent.parent
_REPO_DIR = _PAPER_DIR.parent
_SECTION1_DIR = _REPO_DIR / "benchmark" / "results" / "plan5_quality_section1"
_FIGURES_DIR = _PAPER_DIR / "figures"
_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Make project source importable so we can reuse token_identity_rate.
sys.path.insert(0, str(_REPO_DIR / "src"))
from skillcacher.bench.output_capture import Generation
from skillcacher.bench.output_compare import token_identity_rate


# Academic figure look — small, no spurious decoration
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "pdf.fonttype": 42,  # editable text in vector PDF
    "ps.fonttype": 42,
})


def _gen_from(row: dict) -> Generation:
    blocks = json.loads(row["content_blocks_json"]) if row["content_blocks_json"] else []
    return Generation(
        text=row["text"], content_blocks=blocks,
        stop_reason=row["stop_reason"],
        input_tokens=row["input_tokens"], output_tokens=row["output_tokens"],
        response_id=row["response_id"], ttft_ms=row["ttft_ms"],
    )


def _index(parquet_path: Path) -> dict[tuple[str, int], dict]:
    rows = pq.read_table(parquet_path).to_pylist()
    out = {}
    for r in rows:
        if r["stop_reason"].startswith("replay_error"):
            continue
        out[(r["capture_id"], r["turn_index"])] = r
    return out


def figure_identity_distribution() -> Path:
    """Histogram + CDF of NC↔CB token-identity across §1 corpus.

    Substantive turns (input_tokens ≥ 256) only — the 7 preflight turns
    are excluded since cacheblend's blend_min_tokens=256 means they
    don't exercise the rescue path."""
    nc = _index(_SECTION1_DIR / "no_cache.parquet")
    cb = _index(_SECTION1_DIR / "cacheblend.parquet")
    rates_nc_cb = []
    for key in sorted(set(nc.keys()) & set(cb.keys())):
        if nc[key]["input_tokens"] < 256:
            continue
        rates_nc_cb.append(token_identity_rate(_gen_from(nc[key]), _gen_from(cb[key])))

    rates_nc_cb = np.array(rates_nc_cb)
    median = float(np.median(rates_nc_cb))
    n = len(rates_nc_cb)

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(7.0, 2.4))

    # Histogram (left)
    bins = np.linspace(0, 1.0, 21)
    ax_hist.hist(rates_nc_cb, bins=bins, color="#2c7fb8", edgecolor="white", linewidth=0.4)
    ax_hist.axvline(median, color="firebrick", linestyle="--", linewidth=1.0,
                    label=f"median = {median:.2f}")
    ax_hist.set_xlabel("token_identity_rate(no_cache, cacheblend)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"(a) Distribution (n = {n} substantive turns)")
    ax_hist.legend(loc="upper left", frameon=False)
    ax_hist.set_xlim(-0.02, 1.02)

    # CDF (right) — emphasizes the bimodal shape
    sorted_rates = np.sort(rates_nc_cb)
    cdf = np.arange(1, len(sorted_rates) + 1) / len(sorted_rates)
    ax_cdf.plot(sorted_rates, cdf, color="#2c7fb8", linewidth=1.4)
    ax_cdf.axhline(0.5, color="grey", linestyle=":", linewidth=0.6)
    ax_cdf.axvline(0.3, color="firebrick", linestyle="--", linewidth=0.7,
                   alpha=0.7, label="severe (<0.3)")
    # Shade the severe-divergence region
    p_severe = (sorted_rates < 0.3).mean()
    ax_cdf.fill_between([0, 0.3], [0, 0], [1, 1],
                        color="firebrick", alpha=0.08)
    ax_cdf.set_xlabel("token_identity_rate")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title(f"(b) CDF, {p_severe:.0%} below severe-divergence cutoff")
    ax_cdf.set_xlim(-0.02, 1.02)
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.legend(loc="lower right", frameon=False)

    fig.suptitle(None)
    out = _FIGURES_DIR / "identity_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_judge_summary() -> Path:
    """Stacked bar of judge outcomes — preference for cacheblend vs
    no_cache, with equivalents."""
    csv_path = _SECTION1_DIR / "judge_preferences.csv"
    counts = Counter()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["prefers"] == "equivalent_skipped":
                continue
            counts[row["prefers"]] += 1

    labels = ["prefer_cacheblend", "equivalent", "prefer_no_cache", "unparseable"]
    vals = [counts.get("cacheblend", 0), counts.get("equivalent", 0),
            counts.get("no_cache", 0), counts.get("unparseable", 0)]
    colors = ["#2ca02c", "#bcbd22", "#d62728", "grey"]
    total = sum(vals)

    fig, ax = plt.subplots(figsize=(5.5, 1.6))
    cumulative = 0
    for label, val, color in zip(labels, vals, colors):
        if val == 0:
            continue
        pct = 100 * val / total
        ax.barh(0, val, left=cumulative, color=color, edgecolor="white",
                linewidth=0.5, label=f"{label} ({val}, {pct:.0f}%)")
        # Center label
        ax.text(cumulative + val / 2, 0, f"{val}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        cumulative += val

    ax.set_xlim(0, total)
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_xlabel(f"judgments (n = {total} divergent pairs)")
    ax.set_title(f"Sonnet-4.6 LLM-judge preference, blind A/B (random position)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -2.1),
              ncol=4, frameon=False, fontsize=7.5)

    out = _FIGURES_DIR / "judge_summary.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_per_capture_identity() -> Path:
    """Per-capture per-turn NC↔CB identity strip plot, grouped by subset."""
    nc = _index(_SECTION1_DIR / "no_cache.parquet")
    cb = _index(_SECTION1_DIR / "cacheblend.parquet")
    points = []  # (subset, capture_short, turn, identity, prompt_tokens)
    for key in sorted(set(nc.keys()) & set(cb.keys())):
        capture_id, turn = key
        if nc[key]["input_tokens"] < 256:
            continue
        subset = capture_id.split("/")[0]
        short = capture_id.split("/")[-1]
        if short.startswith("plan4_compaction_spike_v"):
            short = short.replace("plan4_compaction_spike_", "")
        rate = token_identity_rate(_gen_from(nc[key]), _gen_from(cb[key]))
        points.append((subset, short, turn, rate, nc[key]["input_tokens"]))

    captures = sorted({(p[0], p[1]) for p in points})
    cap_to_y = {c: i for i, c in enumerate(captures)}

    fig, ax = plt.subplots(figsize=(7.0, 3.2))
    for subset, short, turn, rate, ptok in points:
        y = cap_to_y[(subset, short)]
        color = "#d62728" if rate < 0.3 else ("#bcbd22" if rate < 0.7 else "#2ca02c")
        ax.scatter(rate, y, s=24, color=color, edgecolor="black", linewidth=0.3,
                   zorder=3)

    ax.set_yticks(range(len(captures)))
    ax.set_yticklabels([f"{s}/{c}" for s, c in captures], fontsize=7)
    ax.set_xlabel("token_identity_rate(no_cache, cacheblend)")
    ax.set_xlim(-0.02, 1.02)
    ax.axvline(0.3, color="firebrick", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.axvline(0.7, color="goldenrod", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_title("Per-capture per-turn identity (substantive turns only)")
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.set_axisbelow(True)

    # Legend
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor="#2ca02c", edgecolor="black", label="$\\geq$0.7"),
        Patch(facecolor="#bcbd22", edgecolor="black", label="0.3--0.7"),
        Patch(facecolor="#d62728", edgecolor="black", label="$<$0.3 (severe)"),
    ], loc="lower left", frameon=False, fontsize=7)

    out = _FIGURES_DIR / "per_capture_identity.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_rescue_distribution() -> Path:
    """Per-turn rescue rate distribution across the full §1 corpus
    (47 substantive turns). Reveals the bimodality: cold turns near 0%
    and steady-state turns near 100%, with a wide intermediate band."""
    csv_path = _SECTION1_DIR / "rescue_per_turn.csv"
    pcts = []
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if int(row["input_tokens"]) < 256:
                continue
            pcts.append(float(row["rescue_pct"]))
    pcts = np.array(pcts)
    n = len(pcts)
    median = float(np.median(pcts))
    mean = float(np.mean(pcts))

    fig, (ax_hist, ax_cdf) = plt.subplots(1, 2, figsize=(7.0, 2.4))
    bins = np.linspace(0, 100, 21)
    ax_hist.hist(pcts, bins=bins, color="#2ca02c", edgecolor="white", linewidth=0.4)
    ax_hist.axvline(median, color="firebrick", linestyle="--", linewidth=1.0,
                    label=f"median = {median:.0f}%")
    ax_hist.axvline(mean, color="black", linestyle=":", linewidth=1.0,
                    label=f"mean = {mean:.0f}%")
    ax_hist.set_xlabel("rescue % per turn (LMCache hit / total prompt)")
    ax_hist.set_ylabel("count")
    ax_hist.set_title(f"(a) Distribution (n = {n} substantive turns)")
    ax_hist.legend(loc="upper center", frameon=False)
    ax_hist.set_xlim(-2, 102)

    sorted_pcts = np.sort(pcts)
    cdf = np.arange(1, n + 1) / n
    ax_cdf.plot(sorted_pcts, cdf, color="#2ca02c", linewidth=1.4)
    ax_cdf.axvline(80, color="goldenrod", linestyle="--", linewidth=0.7, alpha=0.7,
                   label="80% threshold")
    p_cold = (pcts < 1).mean()
    p_warm = (pcts >= 95).mean()
    ax_cdf.fill_between([0, 1], [0, 0], [1, 1], color="firebrick", alpha=0.08)
    ax_cdf.fill_between([95, 100], [0, 0], [1, 1], color="#2ca02c", alpha=0.08)
    ax_cdf.set_xlabel("rescue %")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title(f"(b) CDF: {p_cold:.0%} cold (~0%), {p_warm:.0%} steady (≥95%)")
    ax_cdf.set_xlim(-2, 102)
    ax_cdf.set_ylim(-0.02, 1.02)
    ax_cdf.legend(loc="center right", frameon=False)

    out = _FIGURES_DIR / "rescue_distribution.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_teaser() -> Path:
    """Single-column headline-summary figure for page 1. Two horizontal
    bars stacked vertically: the top bar shows cacheblend's prefill
    rescue rate; the bottom bar shows the LLM-judge preference
    distribution on the divergent pairs. Two findings, one form."""
    csv_path = _SECTION1_DIR / "judge_preferences.csv"
    counts = Counter()
    with csv_path.open() as f:
        for row in csv.DictReader(f):
            if row["prefers"] == "equivalent_skipped":
                continue
            counts[row["prefers"]] += 1
    cb = counts.get("cacheblend", 0)
    eq = counts.get("equivalent", 0)
    nc = counts.get("no_cache", 0)
    total = cb + eq + nc

    # Target: \columnwidth in sigconf is ~3.33in. Make the figure
    # 3.3 x 1.6 so latex doesn't have to rescale.
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(3.3, 1.7),
        gridspec_kw={"height_ratios": [1, 1], "hspace": 1.05})

    # TOP: per-turn rescue rate distribution. Loaded from the same CSV
    # used by figure_rescue_distribution so the teaser's headline number
    # is the actual full-corpus median, not a hand-picked best case.
    rescue_csv = _SECTION1_DIR / "rescue_per_turn.csv"
    rescue_pcts = []
    with rescue_csv.open() as f:
        for row in csv.DictReader(f):
            if int(row["input_tokens"]) < 256:
                continue
            rescue_pcts.append(float(row["rescue_pct"]))
    median_rescue = float(np.median(rescue_pcts))
    n_rescue = len(rescue_pcts)

    # Strip plot of per-turn rescue; vertical median marker.
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.12, 0.12, size=n_rescue)
    ax_top.scatter(rescue_pcts, jitter, s=10, color="#2ca02c", alpha=0.55,
                   edgecolor="none")
    ax_top.axvline(median_rescue, color="firebrick", linestyle="--",
                   linewidth=0.9, zorder=3)
    ax_top.text(median_rescue + 1.5, 0.32, f"median {median_rescue:.0f}%",
                color="firebrick", fontsize=7, fontweight="bold")
    ax_top.set_xlim(-2, 102)
    ax_top.set_ylim(-0.5, 0.5)
    ax_top.set_yticks([])
    ax_top.set_xticks([0, 25, 50, 75, 100])
    ax_top.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=6.5)
    ax_top.tick_params(axis="x", length=2, pad=2)
    ax_top.set_title(f"Rescue: per-turn LMCache hit fraction (n={n_rescue} "
                     f"substantive turns)",
                     fontsize=7.5, pad=3, loc="left")
    for s in ("top", "right", "left"):
        ax_top.spines[s].set_visible(False)
    ax_top.spines["bottom"].set_color("#888")
    ax_top.spines["bottom"].set_linewidth(0.5)

    # BOTTOM: judge preference stacked bar (divergent pairs)
    labels = ["cacheblend", "equivalent", "no_cache"]
    vals = [cb, eq, nc]
    colors = ["#2ca02c", "#bcbd22", "#d62728"]
    cumulative = 0
    for label, val, color in zip(labels, vals, colors):
        if val == 0:
            continue
        pct = 100 * val / total
        ax_bot.barh([0], [val], left=cumulative, color=color,
                    edgecolor="white", height=0.55)
        ax_bot.text(cumulative + val / 2, 0, f"{pct:.0f}%",
                    ha="center", va="center", color="white",
                    fontsize=9, fontweight="bold")
        cumulative += val
    ax_bot.set_xlim(0, total)
    ax_bot.set_ylim(-0.45, 0.45)
    ax_bot.set_yticks([])
    ax_bot.set_xticks([])
    ax_bot.set_title(f"Quality: LLM-judge A/B on $n{{=}}{total}$ divergent pairs",
                     fontsize=7.5, pad=3, loc="left")
    for s in ("top", "right", "left", "bottom"):
        ax_bot.spines[s].set_visible(False)
    # Inline legend below the bar
    ax_bot.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=c, ec="white")
            for c in colors
        ],
        labels=[f"prefer {l}" for l in labels],
        loc="lower center", bbox_to_anchor=(0.5, -0.85),
        ncol=3, frameon=False, fontsize=6.5,
        handlelength=1.0, handleheight=0.7,
        handletextpad=0.35, columnspacing=0.7)

    out = _FIGURES_DIR / "teaser.pdf"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    figures = []
    for fn in (figure_teaser, figure_rescue_distribution,
               figure_identity_distribution,
               figure_judge_summary, figure_per_capture_identity):
        try:
            out = fn()
            figures.append(out)
            print(f"  ✓ {out.relative_to(_REPO_DIR)}")
        except Exception as e:
            print(f"  ✗ {fn.__name__}: {e}")
            raise
    print(f"\nGenerated {len(figures)} figures into {_FIGURES_DIR.relative_to(_REPO_DIR)}/")


if __name__ == "__main__":
    main()
