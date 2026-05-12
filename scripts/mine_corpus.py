"""generic statistical-span miner CLI.

Reads token streams from one of two sources:

  --from-traces <dir>   walk <dir> for ``tokens/req_*.parquet`` (the trace
                        store schema; produced by every capture run).
  --from-raw-requests <dir>   walk <dir> for ``*.json`` Anthropic
                        request bodies (the mtrag fixture shape).
                        Tokenizes via the proxy's ``assemble_and_tokenize``
                        path, mirroring what the proxy sees.

Mines token spans of length ≥ 256 tokens occurring in ≥ 3 distinct
streams, dedups via prefix + MinHash@0.8, and writes one JSONL record
per surviving span. Each record has::

  {"fingerprint": "abc123", "token_ids": [...], "frequency": N,
   "length": M, "source_stream_count": N}

The output JSONL is the input expected by
``SKILLCACHER_STATISTICAL_SPANS_FILE`` on the proxy: each entry triggers
a pre-seed registration with ``source="statistical"``.

Examples:

  # Mine the local fixture corpus
  python -m scripts.mine_corpus --from-traces tests/fixtures/claude_code_real \\
      --out benchmark/results/audit/plan4_miner_local_corpus.jsonl

  # Mine the mtrag workload itself (for the §3 self-ablation)
  python -m scripts.mine_corpus \\
      --from-raw-requests tests/fixtures/mtrag_cloud/mtrag_permuted_extended_30/raw_requests \\
      --out benchmark/results/audit/plan4_miner_mtrag_self.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

from skillcacher.tagging.statistical_miner import mine_spans


def _load_from_traces(root: Path) -> tuple[list[list[int]], list[str]]:
    import pyarrow.parquet as pq
    streams: list[list[int]] = []
    sources: list[str] = []
    for p in sorted(root.rglob("tokens/req_*.parquet")):
        t = pq.read_table(p, columns=["kind", "token_id"])
        d = t.to_pydict()
        prompt_tokens = [
            tid for k, tid in zip(d["kind"], d["token_id"]) if k == "prompt"
        ]
        if not prompt_tokens:
            continue
        streams.append(prompt_tokens)
        # Source label = the first directory name under the traces root.
        try:
            rel = p.relative_to(root)
            sources.append(rel.parts[0])
        except ValueError:
            sources.append("unknown")
    return streams, sources


def _load_from_raw_requests(
    root: Path, tokenizer_name: str,
) -> tuple[list[list[int]], list[str]]:
    """Tokenize Anthropic request bodies the way the proxy does. Uses
    ``assemble_and_tokenize`` so the streams match what the proxy hashes
    on real requests — mining lands tokens that actually correspond to
    proxy chunk boundaries."""
    from skillcacher.proxy.assemble import assemble_and_tokenize
    streams: list[list[int]] = []
    sources: list[str] = []
    for p in sorted(root.glob("*.json")):
        try:
            body = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        tokens, _tags = assemble_and_tokenize(body, tokenizer_name)
        if tokens:
            streams.append(tokens)
            sources.append(root.name)
    return streams, sources


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--from-traces",
        type=Path,
        help="Walk this dir for tokens/req_*.parquet (trace-store shape).",
    )
    src.add_argument(
        "--from-raw-requests",
        type=Path,
        help="Walk this dir for *.json Anthropic request bodies (mtrag shape).",
    )
    p.add_argument(
        "--out", required=True, type=Path,
        help="Output JSONL path. Parent dirs auto-created.",
    )
    p.add_argument(
        "--tokenizer", default="meta-llama/Llama-3.3-70B-Instruct",
        help="Tokenizer (only used for --from-raw-requests; ignored otherwise).",
    )
    p.add_argument("--length-floor", type=int, default=256)
    p.add_argument("--frequency-floor", type=int, default=3)
    p.add_argument("--jaccard-threshold", type=float, default=0.8)
    args = p.parse_args(argv[1:])

    if args.from_traces:
        streams, sources = _load_from_traces(args.from_traces)
        loaded_from = str(args.from_traces)
    else:
        streams, sources = _load_from_raw_requests(
            args.from_raw_requests, args.tokenizer
        )
        loaded_from = str(args.from_raw_requests)

    if not streams:
        print(f"[mine] no token streams loaded from {loaded_from}", file=sys.stderr)
        return 2
    total = sum(len(s) for s in streams)
    print(
        f"[mine] loaded {len(streams)} streams ({total:,} tokens) "
        f"from {loaded_from}",
        file=sys.stderr,
    )
    print(f"[mine] sources: {Counter(sources)}", file=sys.stderr)

    t0 = time.time()
    spans = mine_spans(
        streams,
        length_floor=args.length_floor,
        frequency_floor=args.frequency_floor,
        jaccard_threshold=args.jaccard_threshold,
    )
    dt = time.time() - t0
    print(
        f"[mine] mined {len(spans)} spans (length≥{args.length_floor}, "
        f"freq≥{args.frequency_floor}, J≥{args.jaccard_threshold}) in "
        f"{dt:.1f}s",
        file=sys.stderr,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for s in spans:
            src_set = sorted({sources[sid] for sid in s.source_stream_ids})
            fh.write(json.dumps({
                "fingerprint": s.fingerprint(),
                "token_ids": list(s.token_ids),
                "length": s.length,
                "frequency": s.frequency,
                "source_stream_ids": sorted(s.source_stream_ids),
                "fixture_sources": src_set,
            }) + "\n")
    print(f"[mine] wrote {args.out} ({len(spans)} entries)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
