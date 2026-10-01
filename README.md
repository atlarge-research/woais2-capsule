# Reproducibility capsule

Regenerates the paper's data figures (Figs. 2-6) from IBM's public dataset LLMFineTuningBench.

## Run

Needs Docker, running. In a terminal:

```sh
curl -L -o woais2.zip https://anonymous.4open.science/api/repo/woais2/zip
mkdir woais2 && cd woais2 && unzip -q ../woais2.zip
docker build -t woais2 .
docker run --rm -v ./out:/capsule/out woais2
```

The figures are in `out/`.

Without Docker: `bash reproduce.sh` (needs Python 3.11 and the font Times New Roman).

## Checklist

1. **Program:** `plot.py` (Python 3.11, matplotlib 3.11.0, pandas 2.3.3).
2. **Data:** [`ibm-research/LLMFineTuningBench`](https://huggingface.co/datasets/ibm-research/LLMFineTuningBench),
   30,920 rows x 38 columns; if the Hub fails, the snapshot `data/ado-sfttrainer.csv`.
3. **Environment:** Docker 27.4, base image `python:3.11.16-slim-trixie`.
4. **Hardware:** none specific; tested on ARM64 (macOS) and x86-64.
5. **Output:**

   | File | Paper |
   |---|---|
   | `00_failure_rates_by_category.pdf` | Figure 2 |
   | `03_performance_vs_batch_size.pdf` | Figure 3 |
   | `08_workload_characteristics.pdf` | Figure 4 |
   | `03_insights_method_scaling.pdf` | Figure 5 |
   | `07_optimization_roi.pdf` | Figure 6 |

6. **Disk:** about 1 GB (Docker image).
7. **Time:** about 2 minutes to build, under 1 minute to run.
8. **Data license:** Apache-2.0, IBM Research.
