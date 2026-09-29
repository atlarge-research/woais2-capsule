# Reproducibility capsule

Regenerates the paper's five figures (Figs. 2-6) from IBM's public LLMFineTuningBench dataset.

**Run** `./reproduce.sh`. It needs Python 3.11 and the font Times New Roman (macOS ships it; on Debian/Ubuntu,
install `ttf-mscorefonts-installer`), creates a virtual environment with the pinned packages, and runs `plot.py`.

**Output**: the figures, as PDFs under the paper's file names, in `out/`.

**Data**: [`ibm-research/LLMFineTuningBench`](https://huggingface.co/datasets/ibm-research/LLMFineTuningBench) on
Hugging Face, by IBM Research, under the [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) license. If it cannot
be loaded or has changed, `plot.py` uses the snapshot of its file in `data/ado-sfttrainer.csv`, and prints which source
it used.
