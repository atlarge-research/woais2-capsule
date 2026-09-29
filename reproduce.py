"""Regenerate every dataset-derived number, table and figure of the paper from the public dataset.

Writes out/numbers.json (the dataset's identity; every number with the text the paper prints for it; the plotted
values of every figure), out/table1.txt (Table 1) and out/figures/*.pdf (drawn by figures.py, under the paper's
file names). check.py compares them with the paper.

Definitions
- Outcome. Valid: is_valid == 1. An invalid experiment was rejected before launch (never started) if its
  configuration breaks a validation rule of the actuator: the number of GPUs does not divide the total batch size,
  the expert-parallel degree does not divide the number of GPUs, or the number of nodes does not divide the number
  of GPUs. Every other invalid experiment failed at runtime. The failure ratio counts both. The paper counts a
  rejected experiment under the first rule it breaks, in this order (no experiment breaks the third rule).
- Throughput: dataset_tokens_per_second of a valid run (all GPUs of the job together).
- Matched pair: the valid runs of one configuration at two settings of one factor. All other configuration columns,
  the settings recorded only in the identifier (`hidden`) and the protocol (experiment_id without the factor's own
  token) are equal. A pair's ratio divides the median throughputs of its two settings; we report the median over
  pairs, with the number of pairs and the interquartile range.
- Campaigns A and B: the only campaigns that ran multi-node jobs both with and without RoCE (fms-hf-tuning 2.4.0 and
  2.7.1, each a full and a LoRA experiment_id). They disagree, so RoCE and scaling are reported per campaign (and
  the RoCE speedup also pooled, as the paper prints it). A runs without a time limit; B's experiment_ids stop every
  run after 600 s (stop_after_seconds): the paper's runs without a time limit and runs capped at 600 s.
- Weak-scaling efficiency from g to 2g GPUs: throughput(2g) / (2 x throughput(g)) at equal per-GPU batch, each GPU
  count on the fewest 8-GPU nodes. Across nodes (g >= 8), TCP and RoCE are compared on the configurations both ran.
  Pairs whose g-GPU run peaks at >= 99% GPU memory are left out: a memory-starved run speeds up once its memory
  doubles.
- Speedup of an optimization: the matched ratio with / without it. GPU hours saved for the same work: 1 - 1 / speedup.
"""
import hashlib
import json
import re
import shutil
import warnings
from functools import reduce
from pathlib import Path

import numpy as np
import pandas as pd

import figures

warnings.filterwarnings("ignore", message="The 'verbose' keyword in pd.read_csv is deprecated")

# ── data: the public dataset, loaded as its Hub page shows (Use this dataset); the only data source ──────────
import datasets
datasets.disable_progress_bar()         # its row counter is throttled (it stops at 30,000); rows are counted below
from datasets import load_dataset
ds = load_dataset("ibm-research/LLMFineTuningBench")

assert len(ds) == 1, f"expected a single split, got {list(ds)}"
SPLIT = next(iter(ds))
df = ds[SPLIT].to_pandas()


def content_sha256(frame):
    """SHA-256 of the table as text, independent of row and column order: numbers as repr(float),
    missing cells empty, columns sorted by name, rows sorted."""
    cols = sorted(frame.columns)
    cells = [frame[c].map(lambda v: "" if pd.isna(v) else repr(float(v)))
             if pd.api.types.is_numeric_dtype(frame[c])
             else frame[c].map(lambda v: "" if pd.isna(v) else str(v)) for c in cols]
    rows = sorted("\x1f".join(r) for r in zip(*cells))
    return hashlib.sha256("\n".join(["\x1f".join(cols), *rows]).encode()).hexdigest()


DATASET = {"split": SPLIT, "rows": int(df.shape[0]), "columns": int(df.shape[1]), "content_sha256": content_sha256(df),
           "files": [re.sub(r".*/resolve/", "", url)                       # commit/file the Hub resolved
                     for url in sorted(ds[SPLIT].info.download_checksums or {})]}

OUT = Path("out")
shutil.rmtree(OUT, ignore_errors=True)                                     # no output of an earlier run survives
(OUT / "figures").mkdir(parents=True)

TPS, MEM = "dataset_tokens_per_second", "gpu_memory_utilization_peak"
# Label and memory [GB] per GPU type, in Table 1 order (memory: vendor datasheets; the data lack it).
GPU = {"NVIDIA-A100-SXM4-80GB": ("A100-SXM", 80), "NVIDIA-A100-80GB-PCIe": ("A100-PCIe", 80),
       "L40S": ("L40S", 48), "NVIDIA-H100-PCIe": ("H100-PCIe", 80)}
METHOD = {"full": "Full", "lora": "LoRA", "gptq-lora": "GPTQ-LoRA"}
# Total parameters [B] per LLM, from the model cards (mixture-of-experts models: total, not active). The Granite
# 3.0, 3.1 and 3.3 8B models share one architecture; their cards give 8.1B.
PARAMS_B = {
    "allam-1-13b": 13, "granite-13b-v2": 13, "granite-20b-v2": 20, "granite-3-8b": 8.1, "granite-3.1-2b": 2.5,
    "granite-3.1-3b-a800m-instruct": 3.3, "granite-3.1-8b-instruct": 8.1, "granite-3.3-8b": 8.1,
    "granite-34b-code-base": 34, "granite-3b-code-base-128k": 3, "granite-4.0-1b": 1.6, "granite-4.0-350m": 0.35,
    "granite-4.0-h-1b": 1.5, "granite-4.0-h-micro": 3, "granite-4.0-h-small": 32, "granite-4.0-h-tiny": 7,
    "granite-4.0-micro": 3, "granite-7b-base": 7, "granite-8b-code-base": 8, "granite-8b-japanese": 8,
    "llama-13b": 13, "llama-7b": 6.7, "llama2-70b": 70, "llama3-70b": 70, "llama3-8b": 8, "llama3.1-405b": 405,
    "llama3.1-70b": 70, "llama3.1-8b": 8, "llama3.2-1b": 1.2, "llama3.2-3b": 3.2, "mistral-123b-v2": 123,
    "mistral-7b-v0.1": 7.2, "mixtral-8x7b-instruct-v0.1": 46.7}
MOE = {"granite-3.1-3b-a800m-instruct", "granite-4.0-h-small", "granite-4.0-h-tiny", "mixtral-8x7b-instruct-v0.1"}
assert set(df.model_name) == set(PARAMS_B), set(df.model_name) ^ set(PARAMS_B)
assert set(df.model_name[df.fast_moe > 0]) == MOE, "Fast MoE ran on other LLMs than the MoE ones"

# ── derived columns ─────────────────────────────────────────────────────────────────────────────────────────
assert (df.number_gpus >= 1).all() and set(df.gpu_model) == set(GPU)
df["gpu"] = df.gpu_model.map(lambda g: GPU[g][0])
ep = df.fast_moe.fillna(0)
batch_rule = df.batch_size % df.number_gpus != 0                           # the GPUs do not divide the total batch
ep_rule = (ep > 0) & (df.number_gpus % ep.where(ep > 0, 1) != 0)          # the EP degree does not divide the GPUs
node_rule = df.number_gpus % df.number_nodes != 0                          # the nodes do not divide the GPUs
rejected = batch_rule | ep_rule | node_rule
valid = df.is_valid == 1
assert not (valid & rejected).any(), "a valid run breaks a validation rule"
df["outcome"] = np.select([valid, rejected], ["valid", "rejected"], "runtime")

# Settings recorded only in the identifier ('key.value' pairs joined by '-'), as one matching key.
# HYBRID_SHARD on one node is FULL_SHARD (the actuator documents them as equivalent there).
ID_KEYS = ("dataset_id", "model_name", "number_gpus", "model_max_length", "torch_dtype", "batch_size",
           "gpu_model", "number_nodes", "fast_moe", "enable_roce", "fsdp_sharding_strategy",
           "fsdp_state_dict_type", "fsdp_use_orig_params", "accelerate_config_mixed_precision",
           "gradient_accumulation_steps", "optim", "gradient_checkpointing_use_reentrant", "fast_kernels", "r",
           "lora_alpha", "distributed_backend")
HIDDEN = ("fsdp_sharding_strategy", "fsdp_use_orig_params", "accelerate_config_mixed_precision",
          "gradient_accumulation_steps", "optim", "gradient_checkpointing_use_reentrant", "dataset_id")
KEY_START = re.compile(r"-(?=(?:%s)\.)" % "|".join(ID_KEYS))                # a '-' that starts a known key


def hidden(identifier, nodes):
    kv = dict(part.split(".", 1) for part in KEY_START.split(identifier) if "." in part)
    h = [kv.get(k, "NA") for k in HIDDEN]
    if h[0] in ("NA", "FULL_SHARD") or (h[0] == "HYBRID_SHARD" and nodes == 1):
        h[0] = "FULL"
    return "|".join(h)


df["hidden"] = [hidden(i, n) for i, n in zip(df.identifier, df.number_nodes)]
V = df[df.outcome == "valid"]

NUM = {}


def plain(v):
    """JSON value; floats to 4 decimals, so the record does not depend on the last bits of a platform."""
    if isinstance(v, dict):
        return {plain(k): plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [plain(x) for x in v]
    v = v.item() if hasattr(v, "item") else v
    return round(v, 4) if isinstance(v, float) else v


def put(name, value, text, meaning, **stats):
    """Record a number with the text the paper prints for it (same rounding)."""
    assert re.fullmatch(r"\w+", name) and name not in NUM, name
    NUM[name] = {"value": plain(value), "text": text, "meaning": meaning, **plain(stats)}


def spread(r):
    """Number of pairs and interquartile range of a set of matched ratios."""
    return {"pairs": len(r), "iqr": [r.quantile(0.25), r.quantile(0.75)]}


def span_text(lo, hi, fmt="{}", sep="-"):
    """'lo-hi' as the paper prints a range, or one value if both print alike."""
    a, b = fmt.format(lo), fmt.format(hi)
    return a if a == b else f"{a}{sep}{b}"


def word(label):
    return re.sub(r"\W", "", label)


def pow2_range(values):
    v = sorted({int(x) for x in values})
    a, b = int(np.log2(v[0])), int(np.log2(v[-1]))
    return f"2^{a}-2^{b}" if v == [2 ** k for k in range(a, b + 1)] else ", ".join(map(str, v))


def span(values):
    lo, hi = int(values.min()), int(values.max())
    return f"{lo}–{hi}" if lo < hi else str(lo)


def gpu_types(mask):
    return ", ".join(lab for g, (lab, _) in GPU.items() if (mask & (df.gpu_model == g)).any())


def outcome_shares(d):
    """Valid, rejected and runtime-failure shares [%] of d with one decimal, adding up to exactly 100.0
    (largest-remainder rounding, as Table 1 states)."""
    tenths = 1000 * d.outcome.value_counts().reindex(["valid", "rejected", "runtime"], fill_value=0) / len(d)
    out = np.floor(tenths.to_numpy())
    out[np.argsort(out - tenths.to_numpy(), kind="stable")[:int(round(1000 - out.sum()))]] += 1
    return out / 10


WORD = {3: "three", 4: "four", 7: "seven"}

# ── Abstract, Sec. 1, Sec. 2, Fig. 1: the dataset ─────────────────────────────────────────────────────────────
params = df.model_name.drop_duplicates().map(PARAMS_B)
sxm = 100 * (df.gpu == "A100-SXM").mean()
put("NRuns", len(df), f"{len(df):,}", "experiments (rows)")
put("NModels", df.model_name.nunique(), str(df.model_name.nunique()), "distinct LLMs")
put("NGpuTypes", df.gpu_model.nunique(), WORD[df.gpu_model.nunique()], "GPU types")
put("GpuRange", [df.number_gpus.min(), df.number_gpus.max()],
    f"{df.number_gpus.min():.0f} to {df.number_gpus.max():.0f}", "GPUs per job, fewest to most")
put("ModelParams", [params.min(), params.max()], f"{params.min():g}B-{params.max():g}B",
    "smallest and largest LLM [parameters], from the model cards")
put("NMethods", df.method.nunique(), str(df.method.nunique()), "fine-tuning methods")
put("BatchSizes", pow2_range(df.batch_size), pow2_range(df.batch_size), "total batch sizes [samples]")
put("SeqLengths", pow2_range(df.tokens_per_sample), pow2_range(df.tokens_per_sample), "tokens per sample")
put("NExperimentIds", df.experiment_id.nunique(), str(df.experiment_id.nunique()),
    "experiments as the dataset identifies them (distinct experiment_id)")
assert not (batch_rule & valid).any() and not (ep_rule & valid).any() and not node_rule.any()
n_batch, n_ep = int(batch_rule.sum()), int((ep_rule & ~batch_rule).sum())
put("NRejectedBatchRule", n_batch, f"{n_batch:,}", "experiments rejected before launch: the GPUs do not divide the "
    "total batch size")
put("NRejectedEpRule", n_ep, f"{n_ep:,}", "experiments rejected before launch: the expert-parallel degree does not "
    "divide the GPUs (and the batch rule holds)", breaking_the_rule=int(ep_rule.sum()))
put("PctRowsSxm", sxm, f"{sxm:.0f}%", "share of the experiments on A100-SXM [%]")
OPTIMIZATIONS = {"Fast Kernels": df.fast_kernels.notna(), "Fast MoE": df.fast_moe > 0, "RoCE": df.enable_roce == 1}
n_opt = sum(on.any() and not on.all() for on in OPTIMIZATIONS.values())
put("NOptimizations", n_opt, WORD[n_opt], "optimizations that the experiments switch on and off (Fast Kernels, "
    "Fast MoE, RoCE)")
only = " | ".join(gpu_types(m) for m in (df.number_nodes > 1, df.enable_roce == 1, df.method == "gptq-lora"))
put("OnlySxm", only, only, "GPU types with multi-node jobs | with RoCE | with GPTQ-LoRA")

# ── Table 1: one row per GPU type, the dataset row, and notes a and b ──────────────────────────────────────────
TABLE1 = [["GPU type", "Mem. [GB]", "Exp. [#]", "Ran [%]", "Never started [%]", "Failed at runtime [%]",
           "LLMs", "GPUs", "Nodes"]]
for g, label, mem in [*((g, lab, str(mem)) for g, (lab, mem) in GPU.items()), (None, "Dataset", "")]:
    d = df if g is None else df[df.gpu_model == g]
    ran, never, failed = outcome_shares(d)
    row = [label, mem, f"{len(d):,}", f"{ran:.1f}", f"{never:.1f}", f"{failed:.1f}", str(d.model_name.nunique()),
           span(d.number_gpus), span(d.number_nodes)]
    TABLE1.append(row)
    put("Table1" + word(label), row, " | ".join(row), f"Table 1, {label}: " + "; ".join(TABLE1[0][1:]))
power = df.gpu_power_watts_avg.notna()
no_power = ", ".join(lab for g, (lab, _) in GPU.items() if not power[df.gpu_model == g].any())
put("Table1NoteB", no_power, no_power, "GPU types without GPU power telemetry in any experiment")

# ── Sec. 3.1 and Fig. 2: failures ───────────────────────────────────────────────────────────────────────────────
n = df.outcome.value_counts()
invalid = int(n.rejected + n.runtime)
pinv = 100 * invalid / len(df)
put("NInvalid", invalid, f"{invalid:,}",
    "experiments without a valid run (rejected before launch or failed at runtime)")
put("PctInvalid", pinv, f"{pinv:.1f}%", "failure ratio of all experiments [%]")
put("PctInvalidInt", pinv, f"{pinv:.0f}%", "failure ratio of all experiments [%], to whole percent")
put("InvalidToValid", [pinv, 100 - pinv], f"{pinv:.0f}:{100 - pinv:.0f}", "failure ratio : success ratio [%]")
put("NRejected", n.rejected, f"{n.rejected:,}", "experiments rejected before launch")
put("NRuntime", n.runtime, f"{n.runtime:,}", "experiments that failed at runtime")


def failure_table(key):
    """Experiments, valid ones, failing ones (rejected + runtime) and the failure ratio [%] per value of key."""
    c = pd.crosstab(key, df.outcome).reindex(columns=["valid", "rejected", "runtime"], fill_value=0)
    t = pd.DataFrame({"runs": c.sum(axis=1), "valid": c.valid, "failing": c.rejected + c.runtime,
                      "rejected": c.rejected, "runtime": c.runtime})
    t["failure_pct"] = 100 * t.failing / t.runs
    return t


FAIL = {"batch": failure_table(df.batch_size.astype(int)), "seq": failure_table(df.tokens_per_sample.astype(int)),
        "gpu": failure_table(df.gpu), "method": failure_table(df.method.map(METHOD))}
for b in (1, 2, 4, 8, 16, 1024):
    r = FAIL["batch"].loc[b]
    put(f"FailBatch{b}", r.failure_pct, f"{r.failure_pct:.0f}%", f"failure ratio at total batch size {b} [%]")
mid = 100 * (df.outcome[df.batch_size.between(16, 64)] != "valid").mean()
put("FailPctBatchMid", mid, f"{mid:.0f}%", "failure ratio at total batch sizes 16-64, pooled [%]")
for g in ("L40S", "A100-PCIe"):
    r = FAIL["gpu"].loc[g]
    put(f"FailGpu{word(g)}", [r.failure_pct, int(r.failing)], f"{r.failure_pct:.0f}% ({int(r.failing):,})",
        f"failure ratio of the {g} experiments [%] (failing experiments)")
    put(f"NRuns{word(g)}", int(r.runs), f"{int(r.runs):,}", f"experiments on {g}")
r = FAIL["gpu"].loc["L40S"]
put("RuntimePctL40S", 100 * r.runtime / r.runs, f"{100 * r.runtime / r.runs:.0f}%",
    "share of the L40S experiments that failed at runtime [%]")
put("FailMethodFull", FAIL["method"].loc["Full"].failure_pct, f"{FAIL['method'].loc['Full'].failure_pct:.0f}%",
    "failure ratio of full fine-tuning [%]")
for m in ("LoRA", "Full", "GPTQ-LoRA"):
    r = FAIL["method"].loc[m]
    put(f"RuntimePct{word(m)}", 100 * r.runtime / r.runs, f"{100 * r.runtime / r.runs:.0f}%",
        f"share of the {m} experiments that failed at runtime [%] (of all its experiments)")

# Fig. 2: per panel, bottom to top (batch sizes in order, the other panels by increasing failure ratio)
FIG2_LABEL = {"batch": str, "seq": str,
              "gpu": {"A100-SXM": "A100\nSXM", "A100-PCIe": "A100\nPCIe", "H100-PCIe": "H100\nPCIe",
                      "L40S": "L40S"}.get,
              "method": {"Full": "Full", "LoRA": "LoRA", "GPTQ-LoRA": "GPTQ"}.get}
FIG2 = {}
for panel, t in FAIL.items():
    if panel != "batch":
        t = t.iloc[np.argsort((1 - t.valid.to_numpy() / t.runs.to_numpy()) * 100, kind="stable")]
    FIG2[panel] = {"label": [FIG2_LABEL[panel](k) for k in t.index], "runs": t.runs.astype(int).tolist(),
                   "failing": t.failing.astype(int).tolist(), "rejected": t.rejected.astype(int).tolist()}
PANEL = {"batch": ("Fig2a", "Batch"), "seq": ("Fig2b", "Seq"), "gpu": ("Fig2c", ""), "method": ("Fig2d", "")}
for panel, d in FIG2.items():
    for label, runs, failing in zip(d["label"], d["runs"], d["failing"]):
        rate = (1 - (runs - failing) / runs) * 100                              # as the bar label computes it
        fig, prefix = PANEL[panel]
        put(f"{fig}_{prefix}{word(label)}", [rate, failing], f"{rate:.0f}% ({failing:,})",
            f"Fig. 2, {panel} = {label.replace(chr(10), '-')}: failure ratio [%] (failing experiments)")
for panel in ("batch", "seq"):
    for key, r in FAIL[panel].iterrows():
        split = 100 * r[["failing", "rejected", "runtime"]].to_numpy(float) / r.runs
        put(f"FailSplit{panel.title()}{key}", split,
            "failing {:.1f}%, rejected {:.1f}%, runtime {:.1f}%".format(*split),
            f"Fig. 2, {panel} = {key}: failing = rejected before launch + failed at runtime [% of its experiments]",
            counts=[int(r.failing), int(r.rejected), int(r.runtime), int(r.runs)])
sq = FAIL["seq"]                                                   # 512 to 8,192 tokens per sample
fs, rs, js = (100 * sq[c] / sq.runs for c in ("failing", "runtime", "rejected"))
put("FailSeqRange", [fs.iloc[0], fs.iloc[-1]], f"{fs.iloc[0]:.0f}% → {fs.iloc[-1]:.0f}%",
    "failure ratio at the shortest and at the longest sequence length (512 and 8,192 tokens per sample) [%]")
put("RuntimeSeqRange", [rs.iloc[0], rs.iloc[-1]], f"{rs.iloc[0]:.0f}% to {rs.iloc[-1]:.0f}%",
    "share of the experiments that failed at runtime, at 512 and at 8,192 tokens per sample [%]")
put("RejectedSeqRange", [js.min(), js.max()], span_text(js.min(), js.max(), "{:.0f}") + "%",
    "share of the experiments rejected before launch, lowest to highest over the five sequence lengths [%]")
fm = FAIL["method"].failure_pct.reindex(list(METHOD.values()))
put("FailMethodAll", fm.tolist(), ", ".join(f"{m} {v:.0f}%" for m, v in fm.items()),
    "failure ratio per fine-tuning method [%] (the text prints Full's; Fig. 2d prints all three)")

# ── matched pairs ─────────────────────────────────────────────────────────────────────────────────────────────
CONFIG = ["model_name", "method", "gpu_model", "number_gpus", "number_nodes", "tokens_per_sample", "batch_size",
          "enable_roce", "fast_moe", "fast_kernels", "fms_hf_tuning_version", "torch_dtype", "hidden"]
TOKENS = {  # tokens that encode a factor inside experiment_id; removed so the rest of the protocol matches
    "method": [r"finetune_(?:full|lora|gptq-lora)_benchmark", r"finetune-(?:full|lora|gptq-lora)-"],
    "enable_roce": [r"-enable_roce\.[^-]+"],
    "fast_kernels": [r"-fast_kernels\.\[[^\]]*\]"],
    "number_gpus": [r"-number_nodes\.[^-]+", r"-enable_roce\.[^-]+"],   # RoCE stays in the key as a column
}


def protocol(exp_id, factor):
    for pattern in TOKENS.get(factor, []):
        exp_id = re.sub(pattern, "", exp_id)
    return exp_id


def matched(d, factor, a, b, drop=(), metric=TPS):
    """Median metric of the arms factor == a and factor == b (columns a, b), one row per matched pair."""
    d = d[d.outcome == "valid"].copy()
    for c in ("fast_kernels", "fms_hf_tuning_version", "torch_dtype"):
        d[c] = d[c].fillna("none").astype(str)
    d[["fast_moe", "enable_roce"]] = d[["fast_moe", "enable_roce"]].fillna(0)
    d["protocol"] = d.experiment_id.map(lambda e: protocol(e, factor))
    keys = [c for c in CONFIG if c != factor and c not in drop] + ["protocol"]
    arm = lambda v: d[d[factor] == v].groupby(keys, dropna=False)[metric].median()
    return pd.concat([arm(a).rename("a"), arm(b).rename("b")], axis=1, join="inner")


def speedup(d, factor, a, b):
    j = matched(d, factor, a, b)
    return j.b / j.a


def level(r, key):
    """The value of one configuration column for each pair of r."""
    return r.index.get_level_values(key)


# ── Sec. 3.2 and Figs. 3 and 4: throughput ─────────────────────────────────────────────────────────────────────
ONE = V[V.number_gpus == 1]
FIG3 = {}                                                   # GPU type -> median throughput per batch size, 1 GPU
for g, (lab, _) in GPU.items():
    s = ONE[ONE.gpu_model == g].groupby("batch_size")[TPS].agg(["median", "size"])
    FIG3[lab] = {"batch_size": s.index.astype(int).tolist(), "median_tps": s["median"].tolist(),
                 "runs": s["size"].astype(int).tolist()}
put("NValidOneGpu", len(ONE), f"{len(ONE):,}", "valid runs on one GPU (the data of Fig. 3)")
FIT = {}
for lab, d in FIG3.items():                    # behind 'throughput grows logarithmically with the batch size'
    x, y = np.log2(d["batch_size"]), np.array(d["median_tps"])
    FIT[lab] = slope, r2 = np.polyfit(x, y, 1)[0], np.corrcoef(x, y)[0, 1] ** 2
    put("LogFit" + word(lab), [r2, slope], f"R² {r2:.2f}, {slope:+,.0f} tokens/s per doubling",
        f"least-squares fit of Fig. 3's median throughput on log2(batch size), {lab}, 1 GPU: R², slope")
    one = df[(df.gpu == lab) & (df.number_gpus == 1)]
    steps = {b: speedup(one, "batch_size", b, 2 * b) for b in (1, 2, 4, 8, 16, 32, 64)}
    steps = {b: r for b, r in steps.items() if len(r)}
    put("BatchDoubling" + word(lab), [r.median() for r in steps.values()],
        ", ".join(f"{b}→{2 * b}: {r.median():.2f}×" for b, r in steps.items()),
        f"throughput ratio of matched pairs per batch-size doubling, {lab}, 1 GPU (Fig. 3 compares unmatched "
        "medians, whose mix of LLMs and sequence lengths changes with the batch size)",
        pairs=[len(r) for r in steps.values()])
slope = FIT["A100-SXM"][0]
put("BatchGainPerDoubling", slope, f"about {round(slope, -3):,.0f} tokens/s",
    "throughput added per batch-size doubling on one A100-SXM: slope of the log2 fit of Fig. 3 [tokens/s]")
sx = dict(zip(FIG3["A100-SXM"]["batch_size"], FIG3["A100-SXM"]["median_tps"]))
for b in (1, 8, 128):
    put(f"TpsSxmBatch{b}", sx[b], f"{sx[b]:,.0f}", f"median throughput on one A100-SXM at batch size {b} [tokens/s]")
put("MaxBatchOneGpu", ONE.batch_size.max(), f"{ONE.batch_size.max():.0f}",
    "largest total batch size with a valid one-GPU run", by_gpu={lab: max(d["batch_size"]) for lab, d in FIG3.items()})
put("SxmBatch8Over1", sx[8] / sx[1], f"{sx[8] / sx[1]:.1f}×", "Fig. 3, A100-SXM: median throughput at batch 8 / at 1")
put("SxmBatch128Over8", sx[128] / sx[8], f"{sx[128] / sx[8]:.1f}×",
    "Fig. 3, A100-SXM: median throughput at batch 128 / at 8")

PCIE, SXM, H100 = "NVIDIA-A100-80GB-PCIe", "NVIDIA-A100-SXM4-80GB", "NVIDIA-H100-PCIe"
l40s, sxm_pcie = speedup(df, "gpu_model", PCIE, "L40S"), speedup(df, "gpu_model", PCIE, SXM)
put("PairsL40SA100PCIe", len(l40s), f"{len(l40s):,}", "matched pairs L40S vs. A100-PCIe")
put("L40SLowerThroughput", 1 - l40s.median(), f"{100 * (1 - l40s.median()):.0f}%",
    "throughput of L40S below A100-PCIe, median of matched pairs [%]", **spread(l40s))
put("PairsSxmPcie", len(sxm_pcie), f"{len(sxm_pcie):,}", "matched pairs A100-SXM vs. A100-PCIe")
put("SxmHigherThroughput", sxm_pcie.median() - 1, f"{100 * (sxm_pcie.median() - 1):.0f}%",
    "throughput of A100-SXM above A100-PCIe, median of matched pairs [%]", **spread(sxm_pcie))
put("GpuTypeRatios", [l40s.median(), sxm_pcie.median()],
    f"L40S/A100-PCIe {l40s.median():.2f}× ({len(l40s)} pairs), A100-SXM/A100-PCIe {sxm_pcie.median():.2f}× "
    f"({len(sxm_pcie)} pairs)", "throughput ratios of matched pairs between GPU types")
put("L40SSlowerAllPairs", [(l40s < 1).sum(), l40s.max()], f"{(l40s < 1).sum()} of {len(l40s)} pairs below 1× "
    f"(largest {l40s.max():.2f}×)", "matched pairs in which L40S is slower than A100-PCIe")
put("SxmFasterAllPairs", [(sxm_pcie > 1).sum(), sxm_pcie.min()], f"{(sxm_pcie > 1).sum()} of {len(sxm_pcie)} pairs "
    f"above 1× (smallest {sxm_pcie.min():.2f}×)", "matched pairs in which A100-SXM is faster than A100-PCIe")
h100 = {GPU[g][0]: len(matched(df, "gpu_model", g, H100)) for g in GPU if g != H100}
put("NoH100Pairs", list(h100.values()), ", ".join(f"{n} with {lab}" for lab, n in h100.items()),
    "matched pairs of H100-PCIe with each other GPU type")

BATCHES, SEQS = (sorted(int(v) for v in df[c].unique()) for c in ("batch_size", "tokens_per_sample"))
cell = lambda d: [d.batch_size.astype(int), d.tokens_per_sample.astype(int)]
grid = lambda s: s.unstack().reindex(index=BATCHES, columns=SEQS)
med = grid(V.groupby(cell(V))[TPS].median())
n_valid, n_all = grid(V.groupby(cell(V)).size()).fillna(0).astype(int), grid(df.groupby(cell(df)).size()).fillna(0)
FIG4 = {"batch_size": BATCHES, "sequence_length": SEQS,
        "median_tps": [[None if np.isnan(x) else float(x) for x in row] for row in med.to_numpy(float)],
        "valid_runs": n_valid.to_numpy().tolist(), "experiments": n_all.astype(int).to_numpy().tolist()}
assert n_valid.to_numpy().sum() == len(V) and n_all.to_numpy().sum() == len(df), "a run outside the heatmap"
for b in BATCHES:
    for q in SEQS:
        if not np.isnan(med.loc[b, q]):
            put(f"Fig4_b{b}_s{q}", med.loc[b, q], figures.compact(med.loc[b, q]),
                f"Fig. 4, batch size {b}, {q} tokens per sample: median throughput of {n_valid.loc[b, q]} valid runs "
                "[tokens/s]")
at = lambda cells: ", ".join(f"({b:,}, {q:,})" for b, q in cells)
empty = [(b, q) for b in BATCHES for q in SEQS if np.isnan(med.loc[b, q]) and n_all.loc[b, q] > 0]
untested = [(b, q) for b in BATCHES for q in SEQS if n_all.loc[b, q] == 0]
put("Fig4NoValidRun", empty, at(empty), "Fig. 4, blank outlined cells: (batch size, sequence length) with "
    "experiments but no valid run")
put("Fig4NotTested", untested, at(untested), "Fig. 4, hatched cells: (batch size, sequence length) without experiments")
lo, hi = med.stack().idxmin(), med.stack().idxmax()
put("HeatmapMin", med.loc[lo], figures.compact(med.loc[lo]), f"Fig. 4, the smallest cell, ({lo[0]}, {lo[1]}) "
    "[tokens/s]", cell=list(lo))
put("HeatmapMax", med.loc[hi], figures.compact(med.loc[hi]), f"Fig. 4, the largest cell, ({hi[0]}, {hi[1]}) "
    "[tokens/s]", cell=list(hi))
top = V[(V.batch_size == hi[0]) & (V.tokens_per_sample == hi[1])].number_gpus.astype(int)
put("HeatmapMaxGpus", sorted(set(top)), " or ".join(map(str, sorted(set(top)))),
    "GPU counts of the valid runs in the largest cell", runs={int(g): int(k) for g, k in top.value_counts().items()})
put("NValid", len(V), f"{len(V):,}", "valid runs (the data of Fig. 4)")
for name, (b, q), which in (("HeatmapMinCell", lo, "smallest"), ("HeatmapMaxCell", hi, "largest")):
    put(name, [b, q], f"({b:,}, {q:,})", f"Fig. 4: the cell with the {which} median throughput (batch size, "
        "sequence length)")
put("Cell512x8192", [int(n_all.loc[512, 8192]), int(n_valid.loc[512, 8192])],
    f"{int(n_all.loc[512, 8192]):,} experiments, {int(n_valid.loc[512, 8192])} valid",
    "Fig. 4, batch size 512 at 8,192 tokens per sample: experiments, valid runs")

# ── Sec. 3.3 and Fig. 5: scaling ───────────────────────────────────────────────────────────────────────────────
FIG5 = {}                                                   # method -> median throughput per number of GPUs
for m, label in METHOD.items():
    s = V[V.method == m].groupby("number_gpus")[TPS].agg(["median", "size"])
    FIG5[label] = {"number_gpus": s.index.astype(int).tolist(), "median_tps": s["median"].tolist(),
                   "runs": s["size"].astype(int).tolist()}
per_node = df.number_gpus / df.number_nodes
put("GpusPerNode", per_node.max(), f"{per_node.max():.0f}", "GPUs per node: the most GPUs of one job on one node")
split8 = df[(df.number_gpus == 8) & (df.number_nodes == 2)]
put("EightGpusTwoNodes", [len(split8), int((split8.outcome == "valid").sum()), split8.experiment_id.nunique()],
    f"{len(split8)} experiments ({(split8.outcome == 'valid').sum()} valid) in {split8.experiment_id.nunique()} "
    "experiment_id", "experiments with 8 GPUs on 2 nodes (Fig. 5's 8-GPU medians include them)")
at_gpus = lambda g: [FIG5[m]["median_tps"][FIG5[m]["number_gpus"].index(g)]
                     for m in FIG5 if g in FIG5[m]["number_gpus"]]
for g, name in ((1, "One"), (8, "Eight")):
    t = at_gpus(g)
    put(f"Tps{name}GpuRange", t, span_text(round(min(t), -2), round(max(t), -2), "{:,.0f}"),
        f"Fig. 5 at {g} GPU(s): median throughput, lowest to highest method, to hundreds [tokens/s]")
above = V[V.number_gpus > 8]
lora_full = above.model_name.map(PARAMS_B).groupby(above.method).median()
put("Fig5Above8", {g: {m: FIG5[m]["median_tps"][FIG5[m]["number_gpus"].index(g)] for m in ("Full", "LoRA")}
                   for g in (16, 32)},
    "; ".join(f"{g} GPUs: Full {FIG5['Full']['median_tps'][FIG5['Full']['number_gpus'].index(g)]:,.0f}, LoRA "
              f"{FIG5['LoRA']['median_tps'][FIG5['LoRA']['number_gpus'].index(g)]:,.0f}" for g in (16, 32)),
    "Fig. 5 above 8 GPUs: median throughput of full fine-tuning and LoRA [tokens/s]")
put("MedianParamsLoRAAbove8", lora_full["lora"], f"{lora_full['lora']:.2g}B",
    "median LLM size of the valid LoRA runs on more than 8 GPUs [parameters]")
put("MedianParamsFullAbove8", lora_full["full"], f"{lora_full['full']:.2g}B",
    "median LLM size of the valid full fine-tuning runs on more than 8 GPUs [parameters]")
by_method = speedup(df, "method", "full", "lora")
put("LoRAOverFull", by_method.median() - 1, f"{100 * (by_method.median() - 1):.0f}%",
    "throughput of LoRA above full fine-tuning, median of matched pairs [%]", **spread(by_method))
put("PairsLoRAFull", len(by_method), f"{len(by_method):,}", "matched pairs LoRA vs. full fine-tuning")
put("LoRAOverFullRatio", by_method.median(), f"{by_method.median():.2f}× ({len(by_method):,} pairs)",
    "throughput ratio LoRA / full fine-tuning, median of matched pairs")

multi = df.number_nodes > 1
both = df[multi].groupby("experiment_id").enable_roce.nunique() == 2
C = df[df.experiment_id.isin(both[both].index)]
CAMPAIGN = {"A": C[C.fms_hf_tuning_version == "2.4.0"], "B": C[C.fms_hf_tuning_version == "2.7.1"]}
assert C.experiment_id.nunique() == 4 and len(C) == sum(map(len, CAMPAIGN.values()))
LIMIT = {c: D.experiment_id.str.extract(r"-stop_after_seconds\.([\d.]+)", expand=False).astype(float)
         for c, D in CAMPAIGN.items()}                     # [s] the time limit in the experiment_id, if any
assert LIMIT["A"].isna().all() and not C.experiment_id.str.contains("auto_stop").any(), "campaign A has a time limit"
assert LIMIT["B"].notna().all() and LIMIT["B"].nunique() == 1, "campaign B has no single time limit"
VERSION = {c: D.fms_hf_tuning_version.iloc[0] for c, D in CAMPAIGN.items()}
CAP = LIMIT["B"].iloc[0]
put("VersionUncapped", VERSION["A"], VERSION["A"], "fms-hf-tuning version of campaign A (no time limit)")
put("VersionCapped", VERSION["B"], VERSION["B"], "fms-hf-tuning version of campaign B (runs capped)")
put("TimeCap", CAP, f"{CAP:g} s", "time limit of the runs of campaign B (stop_after_seconds in its experiment_ids)")


def weak(d, g, roce=None):
    """Weak-scaling efficiency g -> 2g GPUs, one value per matched configuration (see Definitions)."""
    w = d[d.number_gpus.isin([g, 2 * g]) & (d.number_nodes == np.ceil(d.number_gpus / 8))].copy()
    w["batch_size"] = w.batch_size / w.number_gpus                               # per-GPU batch
    if roce is not None:                # RoCE acts only across nodes: single-node runs join either arm
        w.loc[w.number_nodes == 1, "enable_roce"] = float(roce)
        w = w[w.enable_roce == float(roce)]
    j = matched(w, "number_gpus", g, 2 * g, drop=("number_nodes",))
    peak = matched(w, "number_gpus", g, 2 * g, drop=("number_nodes",), metric=MEM).a
    e = (j.b / j.a / 2)[peak < 99]
    return e if roce is None else e.droplevel("enable_roce")      # then both arms index the same configurations


EFF = {}                                                    # (g, network) -> campaign -> efficiency per pair
for g in (1, 2, 4, 8, 16):
    arms = {"": None} if g < 8 else {"Tcp": False, "Roce": True}
    for c, D in CAMPAIGN.items():
        e = {net: weak(D, g, on) for net, on in arms.items()}
        same = reduce(pd.Index.intersection, [s.index for s in e.values()])      # configurations every arm ran
        for net in arms:
            EFF.setdefault((g, net), {})[c] = e[net].loc[same]
for g in (1, 2, 4):
    eff = EFF[(g, "")]
    put(f"WeakEff{g}to{2 * g}", {VERSION[c]: e.median() for c, e in eff.items()},
        ", ".join(f"{e.median():.2f} ({VERSION[c]}, {len(e)} pairs)" for c, e in eff.items()),
        f"weak-scaling efficiency {g}->{2 * g} GPUs within a node, median of matched pairs, per campaign",
        **{VERSION[c]: spread(e) for c, e in eff.items()})
intra = [EFF[(g, "")][c].median() for g in (1, 2, 4) for c in CAMPAIGN]
put("WeakEffIntraNode", [min(intra), max(intra)], span_text(min(intra), max(intra), "{:.2f}"),
    "weak-scaling efficiency within a node: lowest to highest median over the doublings 1->2, 2->4, 4->8 GPUs and "
    "both campaigns")
a8 = {net: EFF[(8, net)]["A"] for net in ("Tcp", "Roce")}
lo8, hi8 = min(e.median() for e in a8.values()), max(e.median() for e in a8.values())
put("WeakEff8to16Uncapped", [lo8, hi8], span_text(lo8, hi8, "{:.2f}"),
    "weak-scaling efficiency 8->16 GPUs, campaign A (no time limit), without and with RoCE")
put("Gain8to16Uncapped", [2 * lo8, 2 * hi8], span_text(2 * lo8, 2 * hi8, "{:.2f}") + "×",
    "throughput gain 8->16 GPUs at equal per-GPU batch, campaign A (no time limit), without and with RoCE")
put("Eff8to16UncappedArms", {net: e.median() for net, e in a8.items()},
    f"without RoCE {a8['Tcp'].median():.2f} ({2 * a8['Tcp'].median():.2f}×), with RoCE {a8['Roce'].median():.2f} "
    f"({2 * a8['Roce'].median():.2f}×), {len(a8['Tcp'])} pairs each",
    "weak-scaling efficiency (throughput gain) 8->16 GPUs per network, campaign A (no time limit)",
    **{net: spread(e) for net, e in a8.items()})
for net, name in (("Tcp", "without"), ("Roce", "with")):
    e = EFF[(8, net)]["B"]
    put(f"WeakEff8to16Capped{net}", e.median(), f"{e.median():.2f}",
        f"weak-scaling efficiency 8->16 GPUs {name} RoCE, campaign B (runs capped at 600 s)", **spread(e))
    put(f"Gain8to16Capped{net}", 2 * e.median(), f"{2 * e.median():.2f}×",
        f"throughput gain 8->16 GPUs {name} RoCE at equal per-GPU batch, campaign B (runs capped at 600 s)")
for c, name in (("B", "Capped"), ("A", "Uncapped")):
    e = {net: EFF[(16, net)][c] for net in ("Tcp", "Roce")}
    put(f"Gain16to32{name}", {net: 2 * s.median() for net, s in e.items()},
        f"{2 * e['Tcp'].median():.2f}× without RoCE, {2 * e['Roce'].median():.2f}× with RoCE ({len(e['Tcp'])} pairs "
        "each)", f"throughput gain 16->32 GPUs at equal per-GPU batch, campaign {c}",
        **{n: spread(s) for n, s in e.items()})
b16 = {net: EFF[(16, net)]["B"].median() for net in ("Tcp", "Roce")}
for net, name in (("Tcp", "without"), ("Roce", "with")):
    put(f"Gain16to32Capped{net}", 2 * b16[net], f"{2 * b16[net]:.2f}×",
        f"throughput gain 16->32 GPUs {name} RoCE at equal per-GPU batch, campaign B (runs capped at 600 s)")
put("WeakEff16to32Capped", list(b16.values()), span_text(min(b16.values()), max(b16.values()), "{:.2f}"),
    "weak-scaling efficiency 16->32 GPUs, campaign B (runs capped at 600 s), without and with RoCE")

# ── Sec. 3.4 and Fig. 6: optimizations ─────────────────────────────────────────────────────────────────────────
roce = speedup(df[multi], "enable_roce", 0, 1)
version = level(roce, "fms_hf_tuning_version")
assert set(version) == set(VERSION.values()), "RoCE pairs outside campaigns A and B"
ra, rb = roce[version == VERSION["A"]], roce[version == VERSION["B"]]
put("RoceSpeedup", roce.median(), f"{roce.median():.2f}×", "RoCE on/off throughput of multi-node runs, median of "
    "matched pairs, both campaigns pooled", **spread(roce))
put("PairsRoce", len(roce), f"{len(roce):,}", "matched RoCE pairs (multi-node runs)")
nodes = level(roce, "number_nodes")
put("RoceNodes", [nodes.min(), nodes.max()], span_text(int(nodes.min()), int(nodes.max())), "nodes of the RoCE pairs")
put("RoceUncapped", ra.median(), f"{ra.median():.2f}×", "RoCE speedup, campaign A (no time limit)", **spread(ra))
put("RoceUncappedPct", ra.median() - 1, f"{100 * (ra.median() - 1):.0f}%", "RoCE throughput gain, campaign A [%]")
put("PairsRoceUncapped", len(ra), f"{len(ra):,}", "matched RoCE pairs, campaign A")
by_nodes = lambda r: {int(k): s for k, s in r.groupby(level(r, "number_nodes"))}
ra_n, rb_n = by_nodes(ra), by_nodes(rb)
put("RoceUncapped2Nodes", ra_n[2].median() - 1, f"{100 * (ra_n[2].median() - 1):.0f}%",
    "RoCE throughput gain on 2 nodes, campaign A [%]", **spread(ra_n[2]))
ra4 = {k: s for k, s in ra_n[4].groupby(level(ra_n[4], "model_name"))}
GAINING = {"llama3.1-70b": "Llama70B", "mixtral-8x7b-instruct-v0.1": "Mixtral"}   # the two the paper names
for model, name in GAINING.items():
    put(f"RoceUncapped4Nodes{name}", ra4[model].median(), f"{ra4[model].median():.2f}×",
        f"RoCE speedup on 4 nodes, {model}, campaign A", **spread(ra4[model]))
rest = {k: s for k, s in ra4.items() if k not in GAINING}
put("RoceUncapped4Nodes8B", {k: s.median() for k, s in rest.items()},
    ", ".join(f"{k} {s.median():.2f}× ({len(s)} pairs)" for k, s in rest.items()),
    "RoCE speedup on 4 nodes of every other LLM, campaign A (behind 'only Llama-3.1-70B and Mixtral-8x7B gain')")
for name, r in (("Uncapped", ra_n), ("Capped", rb_n)):
    put(f"RoceByNodes{name}", {k: s.median() for k, s in r.items()},
        ", ".join(f"{k} nodes {s.median():.2f}× ({len(s)} pairs)" for k, s in r.items()),
        f"RoCE speedup per number of nodes, campaign {'A' if name == 'Uncapped' else 'B'}")
put("RoceCapped", rb.median(), f"{rb.median():.2f}×", "RoCE speedup, campaign B (runs capped at 600 s)", **spread(rb))
put("RoceCappedQ3", rb.quantile(0.75), f"{rb.quantile(0.75):.1f}×", "RoCE speedup, campaign B: upper quartile")
put("RoceCappedIQR", [rb.quantile(0.25), rb.quantile(0.75)],
    f"{rb.quantile(0.25):.2f}-{rb.quantile(0.75):.2f}× ({len(rb)} pairs)", "RoCE speedup, campaign B: quartiles")
best = max(ra.median(), rb.median())
put("RoceUpTo", best, f"{best:.1f}×", "RoCE speedup of the campaign that gains most (median of matched pairs)")

EP = sorted(int(e) for e in V.fast_moe.dropna().unique() if e > 0)
moe = {e: speedup(df, "fast_moe", 0, e) for e in EP}
assert all(len(r) for r in moe.values()) and set(V.number_gpus[V.fast_moe == 1]) == {1}, "EP=1 on more GPUs"
by_ep = lambda m: {e: r[level(r, "method") == m] for e, r in moe.items()}
moe_full, moe_lora = by_ep("full"), by_ep("lora")
put("FastMoeEp1", moe[1].median(), f"{moe[1].median():.2f}×", "Fast MoE on/off throughput at EP=1 (experts on one "
    "GPU), median of matched pairs, full fine-tuning and LoRA pooled", **spread(moe[1]))
put("PairsFastMoeEp1", len(moe[1]), f"{len(moe[1]):,}", "matched Fast MoE pairs at EP=1, full fine-tuning and LoRA")
best_moe = max(r.median() for r in moe.values())
put("FastMoeUpTo", best_moe, f"{best_moe:.1f}×", "Fast MoE speedup at the best expert-parallel degree")
for e, m, r in ((1, "Full", moe_full[1]), (1, "LoRA", moe_lora[1]), (8, "LoRA", moe_lora[8])):
    put(f"FastMoeEp{e}{m}", r.median(), f"{r.median():.2f}×", f"Fast MoE speedup at EP={e}, {m}", **spread(r))
spread_ep = [moe[e].median() for e in EP if e > 1]
put("FastMoeEp2to8", [min(spread_ep), max(spread_ep)], span_text(min(spread_ep), max(spread_ep), "{:.2f}") + "×",
    "Fast MoE speedup when the experts spread over more than one GPU (EP > 1): lowest to highest, methods pooled")
many = [e for e in EP if e > 1]
put("FastMoeEpDegrees", many, ", ".join(map(str, many[:-1])) + f", and {many[-1]}",
    "expert-parallel degrees above 1 with valid runs")
tried = {int(e): int((df.fast_moe == e).sum()) for e in sorted(df.fast_moe.dropna().unique()) if e > 1 and e not in EP}
put("FastMoeEpNoValidRun", tried, ", ".join(f"EP={e}: {k} experiments" for e, k in tried.items()),
    "expert-parallel degrees tried without any valid run")
put("FastMoePooledByEp", {e: r.median() for e, r in moe.items()},
    ", ".join(f"{r.median():.2f}× (EP={e}, {len(r)} pairs)" for e, r in moe.items()),
    "Fast MoE speedup per expert-parallel degree, full fine-tuning and LoRA pooled")
for m, r in (("Full", moe_full), ("LoRA", moe_lora)):
    s = [r[e].median() for e in many]
    put(f"FastMoe{m}Ep2to8", s, span_text(min(s), max(s), "{:.2f}") + "×", f"Fast MoE speedup of {m} at EP 2-8")

kernels = speedup(df.assign(fast_kernels=np.where(df.fast_kernels.notna(), "on", "off")), "fast_kernels", "off", "on")
q1, q3 = kernels.quantile(0.25), kernels.quantile(0.75)
put("FastKernels", kernels.median(), f"{kernels.median():.2f}×", "Fast Kernels on/off throughput, median of matched "
    "pairs", **spread(kernels))
put("FastKernelsPct", kernels.median() - 1, f"{100 * (kernels.median() - 1):.0f}%", "Fast Kernels throughput gain [%]")
put("PairsFastKernels", len(kernels), f"{len(kernels):,}", "matched Fast Kernels pairs")
fk_models = sorted(set(level(kernels, "model_name")))
put("FastKernelsModels", fk_models, WORD[len(fk_models)], "LLMs of the Fast Kernels pairs")
biggest = max(PARAMS_B[m] for m in fk_models)
put("FastKernelsMaxParams", biggest, f"{biggest:.0f}B", "largest LLM of the Fast Kernels pairs [parameters]")
put("FastKernelsIQR", [q1 - 1, q3 - 1], f"{100 * (q1 - 1):.0f}-{100 * (q3 - 1):.0f}%",
    "throughput gain of the middle half of the Fast Kernels pairs [%]")
put("FastKernelsIQRRatio", [q1, q3], f"{q1:.2f}-{q3:.2f}×", "Fast Kernels speedup: quartiles of the pairs")
put("FastKernelsAbove1", (kernels > 1).mean(), f"{100 * (kernels > 1).mean():.1f}% of {len(kernels):,} pairs above 1×",
    "share of the Fast Kernels pairs that gain")
hours = {"Fast Kernels": 100 * (1 - 1 / kernels.median()), "RoCE, capped": 100 * (1 - 1 / rb.median())}
put("GpuHoursFastKernels", hours["Fast Kernels"], f"{hours['Fast Kernels']:.0f}%",
    "GPU hours saved by Fast Kernels for the same work [%]")
put("GpuHoursRoceCapped", hours["RoCE, capped"], f"{hours['RoCE, capped']:.0f}%",
    "GPU hours saved by RoCE for the same work, campaign B [%]")
put("GpuHoursRange", list(hours.values()), f"{min(hours.values()):.0f}-{max(hours.values()):.0f}%",
    "GPU hours saved: Fast Kernels to RoCE in campaign B [%]")


def bar(group, label, r):
    """One bar of Fig. 6: its matched ratios' count and quartiles."""
    return {"group": group, "label": label, "n": len(r), "p25": float(r.quantile(0.25)), "p50": float(r.median()),
            "p75": float(r.quantile(0.75))}


FIG6 = [bar("RoCE", "no time limit", ra), bar("RoCE", f"{CAP:g} s limit", rb),
        *(bar("Fast MoE", f"{m}, EP {e}", r[e]) for m, r in (("Full", moe_full), ("LoRA", moe_lora)) for e in EP),
        bar("Fast Kernels", "all pairs", kernels)]
NAME6 = {"no time limit": "Fig6a_NoTimeLimit", f"{CAP:g} s limit": f"Fig6a_{CAP:g}sLimit",
         "all pairs": "Fig6c_AllPairs"}
for row in FIG6:
    name = NAME6.get(row["label"]) or "Fig6b_" + row["label"].replace(", ", "_").replace(" ", "")
    put(name, [row["p50"], row["n"]], f"{row['p50']:.2f}× ({row['n']:,})",
        f"Fig. 6, {row['group']}, {row['label']}: median speedup of matched pairs (pairs)")

# ── figures (figures.py): file name -> (drawing function, its data), in the paper's order (Figs. 2-6) ─────────────
# A new or restyled figure is one function in figures.py, one entry here, and its page size and plotted-values
# hash in check.py.
FIGURES = {
    "00_failure_rates_by_category": (figures.failures, FIG2),
    "03_performance_vs_batch_size": (figures.batch, FIG3),
    "08_workload_characteristics": (figures.heatmap, FIG4),
    "03_insights_method_scaling": (figures.scaling, FIG5),
    "07_optimization_roi": (figures.speedups, FIG6),
}
FIGS = {}
for name, (draw, data) in FIGURES.items():
    sha, plotted = figures.render(draw, data, OUT / "figures" / f"{name}.pdf")
    FIGS[name] = {"pdf_sha256": sha, "data": plain(plotted)}

# ── write ─────────────────────────────────────────────────────────────────────────────────────────────────────


def bordered(rows):
    """Plain-text table with borders on the left, on the right and between the columns."""
    width = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    rule = "+" + "+".join("-" * (x + 2) for x in width) + "+"
    line = lambda r: "| " + " | ".join(c.ljust(x) if i == 0 else c.rjust(x)
                                       for i, (c, x) in enumerate(zip(r, width))) + " |"
    return [rule, line(rows[0]), rule, *map(line, rows[1:-1]), rule, line(rows[-1]), rule]


(OUT / "numbers.json").write_text(json.dumps({"dataset": DATASET, "numbers": NUM, "figures": FIGS}, indent=1,
                                             ensure_ascii=False) + "\n", encoding="utf-8")
multi_node, with_roce, gptq = NUM["OnlySxm"]["text"].split(" | ")
(OUT / "table1.txt").write_text("\n".join([
    "Table 1. The dataset per GPU type. Ran, Never started, Failed at runtime: share of the experiments (each row "
    "sums to 100.0).",
    *bordered(TABLE1),
    f"a: GPU types with multi-node jobs: {multi_node}; with RoCE: {with_roce}; with GPTQ-LoRA: {gptq}.",
    f"b: GPU types without GPU power telemetry: {NUM['Table1NoteB']['text']}."]) + "\n", encoding="utf-8")
print(f"{SPLIT}: {DATASET['rows']:,} rows x {DATASET['columns']} columns from {', '.join(DATASET['files'])}")
print(f"wrote {len(NUM)} numbers and the data of {len(FIGS)} figures to out/numbers.json, Table 1 to "
      f"out/table1.txt, the figures to out/figures/")
