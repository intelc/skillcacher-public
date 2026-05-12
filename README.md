# skillcacher

A transparent proxy that integrates [cacheblend](https://github.com/LMCache/LMCache)
(selective KV recompute) with [Claude Code](https://claude.com/claude-code)
agent traffic on a Llama-3.3-70B-Instruct fp8 / vLLM / LMCache backend,
plus the **ClaudeCodeTrace** benchmark of redacted on-wire Claude Code
request bodies.

This is the artifact accompanying the paper:

> **Hit Rate Is Not Output Quality: Characterizing KV-Cache Reuse on Agent Traffic.**
> Yiheng "Intel" Chen, University of Pennsylvania.

## What's in this repo

```
src/skillcacher/         — proxy, bench harness, judge runner, span registry
scripts/                 — operator entry points (run_judge, redact, publish, download)
scripts/dev/             — reproducibility helpers (oneshot pod, recompute probe, ...)
scripts/deploy/          — RunPod provisioning / bootstrap / teardown
tests/                   — pytest suite (unit + structural)
paper/                   — LaTeX sources (acmart sigconf, figures, refs)
```

The dataset itself lives on Hugging Face:
[`intelchen/claudecode-trace`](https://huggingface.co/datasets/intelchen/claudecode-trace),
CC-BY 4.0.

## Quickstart — replay the dataset

```sh
# 1. Pull the public corpus to ./benchmark/data/
python scripts/download_claudecode_trace.py

# 2. Stand up a 2× H100 pod with the patched lmcache backend
bash scripts/deploy/provision.sh

# 3. Replay one condition (no_cache / prefix_cache / cacheblend)
.venv/bin/skillcacher-bench quality-eval \
    --condition cacheblend \
    --captures swebench_verified \
    --output benchmark/results/my-run/

# 4. Tear down the pod when done
bash scripts/deploy/teardown.sh
```

A 2× H100 80GB pod-hour on RunPod SECURE is ~$5; a full quality-eval pass
across the n=99 main corpus runs in ~30 min. See `paper/sections/09-appendix.tex`
for the full reproducibility recipe and exact environment variables.

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

Then:

```sh
cd paper
make figures   # regenerate plots from benchmark/results/
make pdf       # produces main.pdf
```

## License

Code: MIT (see `LICENSE`).
Paper text + figures: CC-BY 4.0.
Dataset: CC-BY 4.0 (on Hugging Face).

## Citation

If you use this work, please cite the paper (see `CITATION.cff`).
