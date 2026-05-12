# skillcacher

Source artifact for:

> **Hit Rate Is Not Output Quality: Characterizing KV-Cache Reuse on Agent Traffic.**
> Yiheng "Intel" Chen, University of Pennsylvania.
> [`skillcacher-paper.pdf`](skillcacher-paper.pdf) — 19 pages, ACM sigconf.

Skillcacher is a transparent proxy that integrates
[cacheblend](https://github.com/LMCache/LMCache) selective KV recompute
with [Claude Code](https://claude.com/claude-code) agent traffic on a
Llama-3.3-70B-Instruct fp8 / vLLM / LMCache backend. The repo contains
the proxy, the bench harness used in §5, the LLM-judge driver, the
RunPod orchestration scripts, the paper sources, and the pytest suite.

The companion dataset, **ClaudeCodeTrace** (13 redacted Claude Code
captures, 182 turns, 411k tokens; CC-BY 4.0), is published separately
at
[`intelchen/claudecode-trace`](https://huggingface.co/datasets/intelchen/claudecode-trace).

## Layout

```
src/skillcacher/   proxy, bench harness, judge runner, span registry
scripts/           operator entry points (run_judge, redact, download)
scripts/dev/       reproducibility helpers (oneshot pod, recompute probe)
scripts/deploy/    RunPod provisioning / bootstrap / teardown
tests/             pytest suite (unit + structural)
paper/             LaTeX sources (acmart sigconf, figures, refs)
```

## Reproducing the paper numbers

```sh
# 1. Pull the public corpus to ./benchmark/data/
python scripts/download_claudecode_trace.py

# 2. Provision a 2× H100 pod with the patched lmcache backend
bash scripts/deploy/provision.sh

# 3. Replay one condition (no_cache / prefix_cache / cacheblend)
.venv/bin/skillcacher-bench quality-eval \
    --condition cacheblend \
    --captures swebench_verified \
    --output benchmark/results/my-run/

# 4. Tear down
bash scripts/deploy/teardown.sh
```

A 2× H100 80GB SXM pod-hour on RunPod SECURE is approximately $5;
one full quality-eval pass across the n=99 main corpus completes in
about 30 minutes. The end-to-end recipe, including all environment
variables and the LMCACHE_BLEND_RECOMPUTE_RATIOS default, is in
`paper/sections/09-appendix.tex`.

## Building the paper

LaTeX toolchain via BasicTeX (~100 MB):

```sh
brew install --cask basictex
sudo tlmgr update --self
sudo tlmgr install latexmk acmart booktabs microtype xcolor \
    pgf preview ifmtarg biblatex-trad ulem subcaption ms tools \
    geometry trimspaces hyperref totpages environ tikzfill \
    pdfcol kvoptions inconsolata mweights ncctools
```

```sh
cd paper
make figures   # regenerate plots from benchmark/results/
make pdf       # produces main.pdf
```

The shipped `skillcacher-paper.pdf` at the repo root is the same
submission-time build.

## License

- Code: MIT (see `LICENSE`).
- Paper text + figures: CC-BY 4.0.
- Dataset: CC-BY 4.0 (Hugging Face).

## Citation

See `CITATION.cff`.
