"""Compute per-condition TTFT distribution from the §1 corpus parquets.

For each (capture, turn) pair in each condition, the bench harness
recorded the Anthropic SDK's reported ttft_ms (time-to-first-token).
This is the prefill-side latency we want to characterize; output-token
generation latency is largely model-bound and not differentiated by
the cache strategy.
"""
from __future__ import annotations

import statistics as st
from pathlib import Path

import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
SECTION1 = REPO / "benchmark/results/plan5_quality_section1"


def main() -> int:
    print(f"{'condition':<14} {'n':>4} {'mean':>7} {'median':>7} {'p25':>7} "
          f"{'p75':>7} {'p95':>7} {'max':>7}")
    print("-" * 64)
    by_cond = {}
    for cond in ("no_cache", "prefix_cache", "cacheblend"):
        t = pq.read_table(SECTION1 / f"{cond}.parquet")
        ttft = [v for v in t["ttft_ms"].to_pylist()
                if v is not None and v > 0]
        ins = t["input_tokens"].to_pylist()
        ttft_subs = [v for v, i in zip(t["ttft_ms"].to_pylist(), ins)
                     if v and v > 0 and i >= 256]
        by_cond[cond] = (ttft, ttft_subs)
        n = len(ttft_subs)
        s = sorted(ttft_subs)
        print(f"{cond + ' (sub)':<14} {n:>4} {st.mean(ttft_subs):>6.0f}ms "
              f"{st.median(ttft_subs):>6.0f}ms {s[n//4]:>6.0f}ms "
              f"{s[3*n//4]:>6.0f}ms {s[int(0.95*n)]:>6.0f}ms "
              f"{max(ttft_subs):>6.0f}ms")

    # Pairwise speedup of cacheblend vs no_cache on substantive turns
    nc_ttft = by_cond["no_cache"][1]
    pc_ttft = by_cond["prefix_cache"][1]
    cb_ttft = by_cond["cacheblend"][1]
    nc_median = st.median(nc_ttft)
    pc_median = st.median(pc_ttft)
    cb_median = st.median(cb_ttft)
    print()
    print(f"== median TTFT speedup ==")
    print(f"  cacheblend vs no_cache:    {nc_median/cb_median:.2f}x  "
          f"({nc_median:.0f}ms -> {cb_median:.0f}ms, "
          f"{100*(nc_median-cb_median)/nc_median:.1f}% reduction)")
    print(f"  prefix_cache vs no_cache:  {nc_median/pc_median:.2f}x  "
          f"({nc_median:.0f}ms -> {pc_median:.0f}ms)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
