# Artifact Appendix

<p align="center">
  <img src="previews/00_failure_rates_by_category.png" width="100%" alt="Failure ratios per batch size, sequence length, GPU model, and fine-tuning method">
</p>
<p align="center">
  <img src="previews/03_performance_vs_batch_size.png" width="24%" alt="Median throughput per batch size on one GPU">
  <img src="previews/08_workload_characteristics.png" width="24%" alt="Median throughput per batch size and sequence length">
  <img src="previews/03_insights_method_scaling.png" width="24%" alt="Median throughput per number of GPUs">
  <img src="previews/07_optimization_roi.png" width="24%" alt="Median speedup of three optimizations">
</p>

*The paper's figures that the capsule regenerates (previews; the full-resolution PDFs go to `out/figures/`).*

## Abstract

This artifact is a small Python capsule that regenerates every dataset-derived number, table and figure of the
paper from the public LLMFineTuningBench dataset (30,920 LLM fine-tuning experiments, on the Hugging Face Hub as
`ibm-research/LLMFineTuningBench`), and checks them against the paper. It needs no GPU: it analyzes the recorded
experiments. One command, `make docker`, builds a container, downloads the dataset, computes all results, and
checks them; without Docker, `make reproducibility` does the same in a local Python virtual environment.

## Artifact check-list (meta-information)

- **Algorithm:** descriptive statistics; matched-pair comparisons (median ratio over pairs of experiments that
  differ only in one factor); weak-scaling efficiency.
- **Program:** Python scripts `reproduce.py`, `figures.py`, `check.py`.
- **Compilation:** none.
- **Transformations:** none.
- **Binary:** none.
- **Model:** none (the capsule trains no model; the dataset records fine-tuning runs of 33 LLMs).
- **Data set:** LLMFineTuningBench, public on the Hugging Face Hub (`ibm-research/LLMFineTuningBench`),
  30,920 rows x 38 columns, downloaded at run time.
- **Run-time environment:** Docker (`python:3.11-slim` image, pinned by digest, with the font Times New Roman
  added), or Linux/macOS with Python 3.11, the pinned packages of `requirements.txt` and Times New Roman (regular
  and bold).
- **Hardware:** any x86-64 or arm64 machine; no GPU.
- **Run-time state:** none; the run is deterministic.
- **Execution:** `make docker` (or `make reproducibility`).
- **Metrics:** failure ratio [%], throughput [tokens/s], weak-scaling efficiency, speedup [x] over matched pairs.
- **Output:** `out/numbers.json`, `out/table1.txt`, `out/figures/*.pdf`, and the exit status of `check.py`.
- **Experiments:** Table 1 and the analyses of Section 3 (Q1-Q4), with their figures.
- **How much disk space required (approximately)?:** about 0.8 GB (the Docker image); the dataset is a 17 MB CSV file.
- **How much time is needed to prepare workflow (approximately)?:** a few minutes (building the image).
- **How much time is needed to complete experiments (approximately)?:** under one minute.
- **Publicly available?:** yes (anonymized for review).
- **Code licenses (if publicly available)?:** set by the authors at publication.
- **Data licenses (if publicly available)?:** Apache-2.0 (the dataset's license on the Hugging Face Hub).
- **Workflow automation framework used?:** GNU Make and Docker.
- **Archived (provide DOI)?:** not yet; a DOI will be provided at publication.

## Description

### How to access

This folder (anonymized for review).

### Hardware dependencies

None beyond a standard laptop with network access: to the Hugging Face Hub (the dataset) and PyPI; building the
Docker image also downloads from Docker Hub, the Debian mirrors and SourceForge (the fonts).

### Software dependencies

Docker; or Python 3.11 with the exact package versions of `requirements.txt` and the font Times New Roman, regular
and bold (macOS ships it; on Debian or Ubuntu, install `ttf-mscorefonts-installer`). `datasets` is pinned to 2.13.2:
the dataset card names its only split `all`, a name that `datasets` >= 2.14 rejects. `matplotlib` is pinned to
3.11.0, the version that drew the paper's figures: 3.10 places their text up to 1.7 pt elsewhere. The Docker image
starts from `python:3.11-slim` pinned by digest (Python 3.11.16, Debian 13) and installs Debian's
`ttf-mscorefonts-installer`; building it accepts the Microsoft core fonts licence (EULA) and downloads the fonts
from SourceForge.

### Data sets

LLMFineTuningBench, loaded in `reproduce.py` exactly as its Hugging Face Hub page shows (*Use this dataset*), and
the only data source:

    from datasets import load_dataset
    ds = load_dataset("ibm-research/LLMFineTuningBench")

`check.py` verifies 30,920 rows x 38 columns and the recorded content hash, so an upstream change fails loudly.

### Definitions

As implemented at the top of `reproduce.py`:

- **Outcome.** An experiment is valid if `is_valid == 1`. An invalid experiment was *rejected before launch* if its
  configuration breaks a validation rule of the actuator that ran the sweep: (1) the total batch size `batch_size` is
  not a multiple of `number_gpus` (5,360 experiments); (2) Fast MoE is on (`fast_moe` > 0, the expert-parallel
  degree) and `number_gpus` is not a multiple of `fast_moe` (3,784 more: 4,080 break this rule, 296 of them also
  rule 1); (3) `number_gpus` is not a multiple of `number_nodes` (no experiment). Every other invalid experiment
  *failed at runtime* (10,395). No valid run breaks a rule. The failure ratio counts both kinds of failure.
- **Throughput:** `dataset_tokens_per_second` of a valid run (all GPUs of the job together).
- **Table 1** gives each outcome's share of a GPU type's experiments with one decimal, rounded by largest remainder
  so that each row sums to 100.0 (plain rounding would give L40S 15.4% never started, not 15.5%).
- **Figs. 3-5** plot medians of valid runs (Fig. 3: runs on one GPU), not matched pairs. Fig. 5's 8-GPU points
  include the 19 valid runs (36 experiments, one `experiment_id`) that split 8 GPUs over 2 nodes; the weak-scaling
  pairs below leave them out.
- **Matched pair:** the valid runs of one configuration at two settings of one factor (e.g. `enable_roce` 0 and 1).
  All other columns of the configuration are equal: `model_name`, `method`, `gpu_model`, `number_gpus`,
  `number_nodes`, `tokens_per_sample`, `batch_size`, `enable_roce`, `fast_moe`, `fast_kernels`,
  `fms_hf_tuning_version` and `torch_dtype` (a missing value counts as a value); the settings recorded only in
  `identifier` (`fsdp_sharding_strategy`, `fsdp_use_orig_params`, `accelerate_config_mixed_precision`,
  `gradient_accumulation_steps`, `optim`, `gradient_checkpointing_use_reentrant`, `dataset_id`; a missing
  sharding strategy and `HYBRID_SHARD` on one node count as `FULL_SHARD`); and the protocol, `experiment_id`
  without the factor's own token (e.g. `-enable_roce.1`). Fast Kernels compares any kernel list with none.
  Repeated measurements: the runs of one configuration at one setting count once, with their median throughput;
  the pair's ratio divides the median at the second setting by the median at the first. A comparison reports the
  median of its pairs' ratios, the number of pairs and the interquartile range (25th to 75th percentile).
- **The two run classes.** Only four `experiment_id`s ran multi-node jobs both with and without RoCE: a full and a
  LoRA one with fms-hf-tuning 2.4.0, whose runs have no time limit, and a full and a LoRA one with 2.7.1, whose
  `experiment_id`s carry `stop_after_seconds.600`: every run stops after 600 s. `fms_hf_tuning_version` and
  `experiment_id` are part of every matching key, so no pair mixes the classes. RoCE speedups (multi-node runs,
  `enable_roce` 1 vs. 0; every pair is a 2.4.0 or a 2.7.1 one) are reported per class and, as the paper also prints
  them, pooled; weak-scaling efficiencies are computed per class, on its two `experiment_id`s.
- **Weak-scaling efficiency** from g to 2g GPUs: throughput(2g) / (2 x throughput(g)), over matched pairs at equal
  batch size per GPU (`batch_size / number_gpus` replaces the total batch size in the key, and `number_nodes` and
  the node and RoCE tokens of `experiment_id` leave it; `enable_roce` stays), each GPU count on the fewest 8-GPU
  nodes. Across nodes (g >= 8), TCP and RoCE are separate arms that share the one-node runs, and only the
  configurations that both arms ran count. Pairs whose g-GPU runs peak at 99% GPU memory or more
  (`gpu_memory_utilization_peak`, median of the runs) are left out.
- **Speedup** of an optimization: the matched ratio with / without it. GPU hours saved for the same work:
  1 - 1 / speedup.

### Models

None.

## Installation

With Docker (recommended), `make docker` builds the image (`docker build -t llm-finetuning-capsule .`) and runs it.
Without Docker, `make reproducibility` creates `.venv/` from `requirements.txt` (use
`make PYTHON=/path/to/python3.11` if Python 3.11 is not on the path).

## Experiment workflow

    make docker            # build the image, then run it as the host user with ./out mounted
    make reproducibility   # the same without Docker: .venv with pinned packages, download, compute, check
    make clean             # remove .venv, out/ and the caches

Both run `reproduce.py` (download the dataset, compute, write `out/`), then `check.py` (compare with the paper).

## Evaluation and expected results

`reproduce.py` writes
- `out/numbers.json`: the dataset's resolved split, shape and content hash; every number with the text the paper
  prints for it (same rounding), with the number of pairs and the interquartile range of matched comparisons; the
  plotted values of each figure;
- `out/table1.txt`: Table 1 with its notes;
- `out/figures/*.pdf`: the figures, under the paper's file names.

`check.py` exits 0 when
- the data have 30,920 rows x 38 columns and the recorded content hash;
- every number the paper prints (`PAPER`, as the paper prints it) is reproduced with the same text;
- the supporting numbers the paper does not print (`SUPPORT`), and every recorded value, pair count and IQR, are
  unchanged;
- every figure was written by this run with the recorded page size and plotted values, in Times New Roman, by
  matplotlib 3.11.0, and draws what the paper's figure file draws: every object of the PDF (the page's drawing
  operators, markers, images, hatch patterns, font widths and character maps) equals the paper's, except the
  embedded font program and its descriptor, which differ between builds of Times New Roman.

With Debian's build of the font (version 2.82, in the Docker image) the PDF files differ from the paper's (drawn on
macOS, font version 5.01) in the embedded font program, the order of the two font objects and, in Fig. 4, how the
heatmap image is compressed. The glyph outlines and advance widths are the same, so the figures render pixel for
pixel like the paper's in poppler, cairo, PDFium and macOS at any resolution; only renderers that run the font's
TrueType hinting instructions at screen resolution (e.g. Ghostscript below about 120 dpi) shade a few pixels of some
letters differently.

Otherwise it exits non-zero and lists each difference from the paper.

## Experiment customization

The definitions (outcome, throughput, matched pair, campaigns, weak-scaling efficiency, speedup) sit at the top of
`reproduce.py`. A new or restyled figure is one function in `figures.py`, one entry in `FIGURES` in
`reproduce.py`, and its page size, plotted-values hash and drawing hash (`drawing()` of its PDF) in `check.py`.

## Notes

- Run `make clean` before sharing this folder: `.venv/` and `.cache/` record absolute paths of the machine.
- The Agg backend of matplotlib is load-bearing: other backends write narrower PDFs than declared.
- Without Times New Roman, matplotlib would draw the figures in another font, so `figures.py` stops instead. After
  installing the font on Linux, run `make clean`: matplotlib caches the list of fonts it found.
- pandas parses about 2% of the dataset's float values (7,270 of 423,248) one or two units in the last place
  differently on macOS on Apple silicon than on Linux (x86-64 and arm64) and on macOS on Intel, so the data's content
  hash has two recorded values; the numbers and figures are the same on all of them.

## Methodology

Submission, reviewing and badging methodology:

- https://www.acm.org/publications/policies/artifact-review-and-badging-current
- https://cTuning.org/ae
