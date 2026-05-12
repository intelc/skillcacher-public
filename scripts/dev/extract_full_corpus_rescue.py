"""Extract per-turn rescue rates from the §1 corpus vllm.log + parquet.

Reads:
  benchmark/results/plan5_quality_section1/cacheblend.parquet
  benchmark/results/plan5_quality_section1/cacheblend/vllm.log

Writes:
  benchmark/results/plan5_quality_section1/rescue_per_turn.csv

The join key is the OpenAI-style response_id (chatcmpl-XXXXXXXXXXXXXXXX);
vllm.log includes a second segment after the dash that we drop. For
requests that appear in vllm.log multiple times (each LMCache request
prints a STORE-side and a LOOKUP-side line; sometimes a per-chunk line
too), we keep the entry with the largest 'hit tokens' value — that's
the operative one.
"""
from __future__ import annotations

import csv
import re
import statistics as st
from pathlib import Path

import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
SECTION1 = REPO / "benchmark/results/plan5_quality_section1"

LOG_RE = re.compile(
    r"Reqid: (chatcmpl-[a-f0-9]+)(?:-[a-f0-9]+)?, "
    r"Total tokens (\d+),.*?LMCache hit tokens: (\d+)"
)


def main() -> int:
    rescue: dict[str, tuple[int, int]] = {}
    with (SECTION1 / "cacheblend" / "vllm.log").open() as f:
        for line in f:
            m = LOG_RE.search(line)
            if not m:
                continue
            rid_prefix = m.group(1)
            cur = (int(m.group(2)), int(m.group(3)))
            prev = rescue.get(rid_prefix)
            if prev is None or cur[1] > prev[1]:
                rescue[rid_prefix] = cur

    t = pq.read_table(SECTION1 / "cacheblend.parquet")
    cols = {c: t[c].to_pylist() for c in
            ("capture_id", "turn_index", "input_tokens", "response_id")}

    out_csv = SECTION1 / "rescue_per_turn.csv"
    rows = []
    for i in range(t.num_rows):
        rid = cols["response_id"][i]
        total, hit = rescue.get(rid, (None, None))
        pct = 100.0 * hit / total if total else 0.0
        rows.append({
            "capture_id": cols["capture_id"][i],
            "turn_index": cols["turn_index"][i],
            "input_tokens": cols["input_tokens"][i],
            "engine_total_tokens": total if total is not None else 0,
            "lmcache_hit_tokens": hit if hit is not None else 0,
            "rescue_pct": round(pct, 2),
        })
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows to {out_csv}")

    # Summary stats over substantive turns (input_tokens >= 256)
    substantive = [r for r in rows if r["input_tokens"] >= 256]
    pcts = [r["rescue_pct"] for r in substantive]
    n = len(pcts)
    sorted_pcts = sorted(pcts)
    print()
    print(f"=== rescue distribution over {n} substantive turns ===")
    print(f"  mean    = {st.mean(pcts):.1f}%")
    print(f"  median  = {st.median(pcts):.1f}%")
    print(f"  p25     = {sorted_pcts[n // 4]:.1f}%")
    print(f"  p75     = {sorted_pcts[3 * n // 4]:.1f}%")
    print(f"  min     = {min(pcts):.1f}%")
    print(f"  max     = {max(pcts):.1f}%")
    print(f"  >=99%:    {sum(1 for p in pcts if p >= 99)}/{n} "
          f"({100*sum(1 for p in pcts if p >= 99)/n:.0f}%)")
    print(f"  >=95%:    {sum(1 for p in pcts if p >= 95)}/{n} "
          f"({100*sum(1 for p in pcts if p >= 95)/n:.0f}%)")
    print(f"  >=80%:    {sum(1 for p in pcts if p >= 80)}/{n} "
          f"({100*sum(1 for p in pcts if p >= 80)/n:.0f}%)")
    print(f"  <50%:     {sum(1 for p in pcts if p < 50)}/{n} "
          f"({100*sum(1 for p in pcts if p < 50)/n:.0f}%)")
    print(f"  <1% (cold): {sum(1 for p in pcts if p < 1)}/{n} "
          f"({100*sum(1 for p in pcts if p < 1)/n:.0f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
