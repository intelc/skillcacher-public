"""Compute valid-tool-call rate per condition + per-pair mode agreement.

For each substantive (capture, turn) pair, classify each condition's
output as one of {tool_call, prose, empty}. CC's wire protocol can only
act on tool-calls; a prose response stalls the agent. The reviewer
specifically asked for an end-to-end agent metric --- this is the
cheapest one we can extract from existing parquets.
"""
from __future__ import annotations

import csv
import re
import sys
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq


REPO = Path(__file__).resolve().parents[2]
SECTION1 = REPO / "benchmark/results/plan5_quality_section1"

# Tool-call shape: text begins (possibly after whitespace) with a JSON
# object that has `"type": "function"` and a `"name"` key. The full
# Claude wire shape can have surrounding structure, but the
# tool-call-as-first-block heuristic is reliable: CC's harness only
# acts on outputs whose first block parses as a tool call.
_TOOL_HEAD = re.compile(
    r'^\s*\{[^{}]*?"type"\s*:\s*"function"[^{}]*?"name"\s*:\s*"[A-Za-z_][A-Za-z0-9_]*"',
    re.DOTALL,
)


def classify(text: str | None) -> str:
    if not text:
        return "empty"
    if _TOOL_HEAD.search(text):
        return "tool_call"
    return "prose"


def main() -> int:
    nc = pq.read_table(SECTION1 / "no_cache.parquet")
    cb = pq.read_table(SECTION1 / "cacheblend.parquet")

    def index(t):
        out = {}
        for i in range(t.num_rows):
            key = (t["capture_id"][i].as_py(), t["turn_index"][i].as_py())
            out[key] = {
                "text": t["text"][i].as_py(),
                "input_tokens": t["input_tokens"][i].as_py(),
                "output_tokens": t["output_tokens"][i].as_py(),
            }
        return out

    nc_rows = index(nc)
    cb_rows = index(cb)

    keys = sorted(set(nc_rows) & set(cb_rows))
    nc_substantive = Counter()
    cb_substantive = Counter()
    pair_mode = Counter()  # (nc_class, cb_class) -> count, substantive only
    detailed = []

    for k in keys:
        n = nc_rows[k]
        c = cb_rows[k]
        if n["input_tokens"] < 256:
            continue
        nc_cls = classify(n["text"])
        cb_cls = classify(c["text"])
        nc_substantive[nc_cls] += 1
        cb_substantive[cb_cls] += 1
        pair_mode[(nc_cls, cb_cls)] += 1
        detailed.append({
            "capture_id": k[0],
            "turn_index": k[1],
            "input_tokens": n["input_tokens"],
            "nc_class": nc_cls,
            "cb_class": cb_cls,
            "nc_output_tokens": n["output_tokens"],
            "cb_output_tokens": c["output_tokens"],
        })

    out_csv = SECTION1 / "tool_call_classification.csv"
    with out_csv.open("w") as f:
        w = csv.DictWriter(f, fieldnames=detailed[0].keys())
        w.writeheader()
        w.writerows(detailed)

    n = len(detailed)
    print(f"== n = {n} substantive pairs ==\n")
    print(f"no_cache:   tool_call={nc_substantive['tool_call']}/{n} "
          f"({100*nc_substantive['tool_call']/n:.1f}%), "
          f"prose={nc_substantive['prose']}/{n} "
          f"({100*nc_substantive['prose']/n:.1f}%), "
          f"empty={nc_substantive['empty']}/{n} "
          f"({100*nc_substantive['empty']/n:.1f}%)")
    print(f"cacheblend: tool_call={cb_substantive['tool_call']}/{n} "
          f"({100*cb_substantive['tool_call']/n:.1f}%), "
          f"prose={cb_substantive['prose']}/{n} "
          f"({100*cb_substantive['prose']/n:.1f}%), "
          f"empty={cb_substantive['empty']}/{n} "
          f"({100*cb_substantive['empty']/n:.1f}%)")

    print(f"\n== pair mode agreement ==")
    print(f"NC | CB     | count | %")
    for (nc_cls, cb_cls), cnt in sorted(pair_mode.items(),
                                        key=lambda x: -x[1]):
        marker = "OK" if nc_cls == cb_cls else "DRIFT"
        print(f"{nc_cls:<10} | {cb_cls:<10} | {cnt:>4d}  | "
              f"{100*cnt/n:5.1f}%  {marker}")

    agree = sum(c for (a, b), c in pair_mode.items() if a == b)
    print(f"\nmode agreement rate: {agree}/{n} ({100*agree/n:.1f}%)")
    # Specifically the cacheblend-introduces-prose case
    nc_tool_cb_prose = pair_mode.get(("tool_call", "prose"), 0)
    nc_prose_cb_tool = pair_mode.get(("prose", "tool_call"), 0)
    print(f"NC tool-call → CB prose:  {nc_tool_cb_prose}/{n} "
          f"({100*nc_tool_cb_prose/n:.1f}%)  ← agent stall")
    print(f"NC prose → CB tool-call:  {nc_prose_cb_tool}/{n} "
          f"({100*nc_prose_cb_tool/n:.1f}%)  ← agent recovery")
    print(f"\nwrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
