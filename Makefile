# `make reproducibility` regenerates every dataset-derived number, table and figure of the paper and checks them
# against the paper. It needs Python 3.11 and the font Times New Roman, regular and bold, in which the figures are set:
# macOS ships it; on Debian or Ubuntu, install ttf-mscorefonts-installer, then run `make clean` (matplotlib caches the
# list of fonts it found).
PYTHON ?= $(shell command -v python3.11)
VENV := .venv
PY := $(VENV)/bin/python

# Agg is load-bearing: other matplotlib backends write narrower PDFs than declared.
# All caches stay inside this folder. `datasets` logs only errors, but still prints its cache folder (a local path)
# when it downloads the dataset; no output file holds a path.
export MPLBACKEND := Agg
export MPLCONFIGDIR := $(CURDIR)/.cache/matplotlib
export HF_HOME := $(CURDIR)/.cache/huggingface
export PIP_CACHE_DIR := $(CURDIR)/.cache/pip
export DATASETS_VERBOSITY := error
export PYTHONDONTWRITEBYTECODE := 1

IMAGE ?= llm-finetuning-capsule

.PHONY: reproducibility docker clean

reproducibility: $(VENV)/.installed
	$(PY) reproduce.py
	$(PY) check.py

# The same run in a container (only Docker needed): build the image, then run it as the host user with ./out mounted.
docker:
	docker build -t $(IMAGE) .
	mkdir -p out
	docker run --rm --user "$$(id -u):$$(id -g)" -v "$(CURDIR)/out:/capsule/out" $(IMAGE)

# The venv is built from the resolved interpreter: one reached through a symlink can lose its stdlib.
$(VENV)/.installed: requirements.txt
	@test -n "$(PYTHON)" || { echo "Needs Python 3.11: make PYTHON=/path/to/python3.11"; exit 1; }
	"$$($(PYTHON) -c 'import os, sys; print(os.path.realpath(sys.executable))')" -m venv $(VENV)
	$(PY) -m pip install --quiet --disable-pip-version-check -r requirements.txt
	touch $@

clean:
	rm -rf $(VENV) out .cache
