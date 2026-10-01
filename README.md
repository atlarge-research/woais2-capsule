# Reproducibility capsule

Regenerates the paper's five data figures (Figs. 2-6) from IBM's public LLMFineTuningBench dataset.

## Run (Docker, recommended)

```sh
docker build -t woais2 .
docker run --rm -v "$PWD/out:/capsule/out" woais2
```

The figures appear in `out/`. The image brings Python 3.11, the font Times New Roman and the pinned packages, so
nothing else needs to be installed. On Linux, add `--user "$(id -u):$(id -g)"` after `--rm` to own the produced files.

## Run (without Docker)

```sh
bash reproduce.sh
```

It needs Python 3.11 (`python3.11` on the PATH, or `PYTHON=/path/to/python3.11 bash reproduce.sh`; e.g., `brew install
python@3.11` or `uv python install 3.11`) and the font Times New Roman (macOS ships it; on Debian/Ubuntu, install
`ttf-mscorefonts-installer`). It creates a virtual environment in `.venv/` with the pinned packages and runs `plot.py`.
Use `bash reproduce.sh`, not `./reproduce.sh`: a downloaded archive may drop the file's executable permission.

## Checklist

1. **Program:** `plot.py` (Python 3.11, matplotlib 3.11.0, pandas 2.3.3).
2. **Data set:** [`ibm-research/LLMFineTuningBench`](https://huggingface.co/datasets/ibm-research/LLMFineTuningBench)
   on Hugging Face (30,920 rows x 38 columns). If it cannot be loaded or has changed, `plot.py` uses the snapshot of its
   file in `data/ado-sfttrainer.csv`, and prints which source it used.
3. **Run-time environment:** Docker (tested on Docker 27.4, `python:3.11.16-slim-trixie` base image, pinned by digest),
   or a local Python 3.11.
4. **Hardware:** no specific hardware; tested on ARM64 (macOS, natively and via Docker) and x86-64 (the `linux/amd64`
   Docker image).
5. **Metrics:** failure ratio, throughput [tokens/s] and speedup, as in Figs. 2-6.
6. **Output:** the figures, in PDF format, in `out/`:

   | File | Paper |
   |---|---|
   | `00_failure_rates_by_category.pdf` | Figure 2 |
   | `03_performance_vs_batch_size.pdf` | Figure 3 |
   | `08_workload_characteristics.pdf` | Figure 4 |
   | `03_insights_method_scaling.pdf` | Figure 5 |
   | `07_optimization_roi.pdf` | Figure 6 |

7. **Expected results:** the figures match the paper's. With Docker, the files differ in bytes from the ones in `out/`
   only because Debian's Times New Roman is another version of the font file; rendered, they are identical.
8. **Disk space:** about 1 GB for the Docker image; about 0.4 GB for `.venv/` without Docker.
9. **Time to prepare:** a few minutes, mostly downloads (about 2 minutes on a fast connection).
10. **Time to run:** under 1 minute (up to about 2 minutes more if a firewall silently drops traffic to Hugging Face,
    before `plot.py` falls back to the snapshot).
11. **Data license:** [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0), by IBM Research.
