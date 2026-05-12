# Skillcacher paper build
#
# Targets:
#   make pdf       — build main.pdf via latexmk (default)
#   make figures   — regenerate figures from benchmark/ data
#   make clean     — remove LaTeX intermediates
#   make distclean — also remove main.pdf and generated figures

LATEXMK ?= latexmk
LATEXMK_FLAGS ?= -pdf -interaction=nonstopmode -file-line-error -shell-escape
PYTHON ?= ../.venv/bin/python

MAIN := main
SECTIONS := $(wildcard sections/*.tex)
BIB := refs.bib

FIGURE_SRC := scripts/make_figures.py
FIGURE_OUT := figures/teaser.pdf \
              figures/rescue_distribution.pdf \
              figures/identity_distribution.pdf \
              figures/judge_summary.pdf \
              figures/per_capture_identity.pdf
# Where the figure script expects the input parquets. Override at the
# command line if your benchmark output lives elsewhere:
#   make figures BENCH_DATA=path/to/results
BENCH_DATA ?= ../benchmark/results/quality_eval

.PHONY: pdf figures clean distclean watch

pdf: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(SECTIONS) $(BIB)
	$(LATEXMK) $(LATEXMK_FLAGS) $(MAIN)

# `make figures` regenerates from $(BENCH_DATA); the prebuilt PDFs shipped
# under figures/ are reproducible from the published ClaudeCodeTrace dataset.
figures:
	$(PYTHON) $(FIGURE_SRC) --bench-data $(BENCH_DATA)

# Watch mode (latexmk -pvc) — auto-rebuild on save
watch:
	$(LATEXMK) $(LATEXMK_FLAGS) -pvc $(MAIN)

clean:
	$(LATEXMK) -c $(MAIN)
	rm -f *.aux *.bbl *.bcf *.blg *.fdb_latexmk *.fls *.log *.out *.run.xml *.synctex.gz *.toc
	rm -f sections/*.aux

distclean: clean
	rm -f $(MAIN).pdf
	rm -f $(FIGURE_OUT)
