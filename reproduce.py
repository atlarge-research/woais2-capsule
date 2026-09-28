"""Regenerate every dataset-derived number, table and figure of the paper from the public dataset.

Writes out/numbers.json (the dataset's identity; every number with the text the paper prints for it; the plotted
values of every figure), out/table1.txt (Table 1) and out/figures/*.pdf (drawn by figures.py, under the paper's
file names). check.py compares them with the paper.

Definitions
- Outcome. Valid: is_valid == 1. An invalid experiment was rejected before launch (never started) if its
  configuration breaks a validation rule of the actuator: the number of GPUs does not divide the total batch size,
  the expert-parallel degree does not divide the number of GPUs, or the number of nodes does not divide the number
  of GPUs. Every other invalid experiment failed at runtime. The failure ratio counts both.
- Throughput: dataset_tokens_per_second of a valid run (all GPUs of the job together).
- Matched pair: the valid runs of one configuration at two settings of one factor. All other configuration columns,
  the settings recorded only in the identifier (`hidden`) and the protocol (experiment_id without the factor's own
  token) are equal. A pair's ratio divides the median throughputs of its two settings; we report the median over
  pairs, with the number of pairs and the interquartile range.
- Campaigns A and B: the only campaigns that ran multi-node jobs both with and without RoCE (fms-hf-tuning 2.4.0 and
  2.7.1, each a full and a LoRA experiment_id). They disagree, so RoCE and scaling are reported per campaign (and
  the RoCE speedup also pooled, as the paper prints it).
- Weak-scaling efficiency from g to 2g GPUs: throughput(2g) / (2 x throughput(g)) at equal per-GPU batch, each GPU
  count on the fewest 8-GPU nodes. Across nodes (g >= 8), TCP and RoCE are compared on the configurations both ran.
  Pairs whose g-GPU run peaks at >= 99% GPU memory are left out: a memory-starved run speeds up once its memory
  doubles.
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

# ── data: the public dataset, loaded as its card shows; the only data source ─────────────────────────────────
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
# Total parameters [B] per LLM, from the model cards (mixture-of-experts models: total, not active).
PARAMS_B = {
    "allam-1-13b": 13, "granite-13b-v2": 13, "granite-20b-v2": 20, "granite-3-8b": 8, "granite-3.1-2b": 2.5,
    "granite-3.1-3b-a800m-instruct": 3.3, "granite-3.1-8b-instruct": 8.1, "granite-3.3-8b": 8.2,
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
rejected = ((df.batch_size % df.number_gpus != 0)
            | ((ep > 0) & (df.number_gpus % ep.where(ep > 0, 1) != 0))
            | (df.number_gpus % df.number_nodes != 0))
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
        return {k: plain(x) for k, x in v.items()}
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


def by_campaign(name, ratios, fmt, meaning):
    """Record the median of each campaign's matched ratios, printed as 'x (A), y (B)'."""
    med = {c: r.median() for c, r in ratios.items()}
    put(name, med, ", ".join(f"{fmt.format(m)} ({c})" for c, m in med.items()), meaning,
        **{c: spread(r) for c, r in ratios.items()})


def word(label):
    return re.sub(r"\W", "", label)


def pow2_range(values):
    v = sorted({int(x) for x in values})
    a, b = int(np.log2(v[0])), int(np.log2(v[-1]))
    return f"2^{a}–2^{b}" if v == [2 ** k for k in range(a, b + 1)] else ", ".join(map(str, v))


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


# ── Abstract, Sec. 1, Sec. 2, Fig. 1: the dataset ─────────────────────────────────────────────────────────────
params = df.model_name.drop_duplicates().map(PARAMS_B)
WORD = {3: "three", 4: "four"}
sxm = 100 * (df.gpu == "A100-SXM").mean()
n = df.outcome.value_counts()
put("NRuns", len(df), f"{len(df):,}", "experiments (rows)")
put("NOutcomes", [n.valid, n.rejected, n.runtime], f"{n.valid:,} valid, {n.rejected:,} rejected, {n.runtime:,} failed "
    f"at runtime ({n.rejected + n.runtime:,} invalid)", "experiments per outcome")
put("NModels", df.model_name.nunique(), str(df.model_name.nunique()), "distinct LLMs")
put("ModelParams", [params.min(), params.max()], f"{params.min():g}B–{params.max():g}B",
    "smallest and largest LLM [parameters], from the model cards")
put("NMethods", df.method.nunique(), WORD[df.method.nunique()], "fine-tuning methods")
put("NGpuTypes", df.gpu_model.nunique(), WORD[df.gpu_model.nunique()], "GPU types")
put("GpuRange", span(df.number_gpus), span(df.number_gpus), "GPUs per job")
put("BatchSizes", pow2_range(df.batch_size), pow2_range(df.batch_size), "total batch sizes [samples]")
put("SeqLengths", pow2_range(df.tokens_per_sample), pow2_range(df.tokens_per_sample), "tokens per sample")
put("NExperimentIds", df.experiment_id.nunique(), str(df.experiment_id.nunique()),
    "experiments (distinct experiment_id)")
put("NValid", len(V), f"{len(V):,}", "valid runs")
put("PctRowsSxm", sxm, f"{sxm:.0f}%", "share of the experiments on A100-SXM [%]")
only = " | ".join(gpu_types(m) for m in (df.number_nodes > 1, df.enable_roce == 1, df.method == "gptq-lora"))
put("OnlySxm", only, only, "GPU types with multi-node jobs | with RoCE | with GPTQ-LoRA")

# ── Table 1: one row per GPU type, a total row, notes, and the configuration-space block ──────────────────────
TABLE1 = [["GPU type", "Mem. [GB]", "Exp. [#]", "Ran [%]", "Never started [%]", "Failed at runtime [%]",
           "LLMs", "GPUs", "Nodes"]]
for g, label, mem in [*((g, lab, str(mem)) for g, (lab, mem) in GPU.items()), (None, "All", "")]:
    d = df if g is None else df[df.gpu_model == g]
    ran, never, failed = outcome_shares(d)
    row = [label, mem, f"{len(d):,}", f"{ran:.1f}", f"{never:.1f}", f"{failed:.1f}", str(d.model_name.nunique()),
           span(d.number_gpus), span(d.number_nodes)]
    TABLE1.append(row)
    put("Table1" + word(label), row, " | ".join(row), f"Table 1, {label}: " + "; ".join(TABLE1[0][1:]))
power = V.gpu_power_watts_avg.notna()
no_power = ", ".join(lab for g, (lab, _) in GPU.items() if not power[V.gpu_model == g].any())
put("Table1NoteB", no_power, f"{no_power}; {power.sum():,} of {len(V):,}",
    "GPU types without power data; valid runs with GPU power, of all valid runs")
m = 100 * df.method.value_counts(normalize=True)[list(METHOD)]
put("MethodShares", m.tolist(), " / ".join(f"{v:.1f}" for v in m), "Full / LoRA / GPTQ-LoRA [% of experiments]")
put("NModelsMoe", len(MOE), str(len(MOE)), "mixture-of-experts LLMs (model cards)")
put("NRunsRoce", (df.enable_roce == 1).sum(), f"{(df.enable_roce == 1).sum():,}", "experiments with RoCE")
put("NVersions", df.fms_hf_tuning_version.nunique(), str(df.fms_hf_tuning_version.nunique()),
    "fms-hf-tuning versions")

# ── Fig. 3: failure ratio per GPU type, method, batch size and sequence length ────────────────────────────────


def failure_table(key, by_ratio=True):
    """Experiments, failing ones (rejected + runtime) and failure ratio [%] per value of key, bottom to top."""
    c = pd.crosstab(key, df.outcome).reindex(columns=["valid", "rejected", "runtime"], fill_value=0)
    t = pd.DataFrame({"runs": c.sum(axis=1), "failing": c.rejected + c.runtime, "rejected": c.rejected,
                      "runtime": c.runtime})
    t["failure_pct"] = 100 * t.failing / t.runs
    t.index = t.index.astype(str)
    return t.sort_values("failure_pct", kind="stable") if by_ratio else t


FAIL = {"gpu": failure_table(df.gpu), "method": failure_table(df.method.map(METHOD)),
        "batch": failure_table(df.batch_size.astype(int), by_ratio=False),      # in batch-size order
        "seq": failure_table(df.tokens_per_sample.astype(int))}
for panel, t in FAIL.items():
    for label, r in t.iterrows():
        put(f"Fail{panel.title()}{word(label)}", r.failure_pct, f"{r.failure_pct:.0f}% ({int(r.failing):,})",
            f"Fig. 3, {panel} = {label}: failure ratio [%] (failing experiments)")
for b, r in FAIL["batch"].iterrows():
    split = 100 * r[["failing", "rejected", "runtime"]].to_numpy(float) / r.runs
    put(f"FailSplitBatch{b}", split, "invalid {:.1f}%, rejected {:.1f}%, runtime {:.1f}%".format(*split),
        f"Fig. 3c, batch size {b}: invalid = rejected before launch + failed at runtime [% of its experiments]",
        counts=[int(r.failing), int(r.rejected), int(r.runtime), int(r.runs)])
mid = 100 * (df.outcome[df.batch_size.between(16, 64)] != "valid").mean()
put("FailPctBatchMid", mid, f"{mid:.0f}%", "failure ratio at total batch sizes 16-64, pooled [%]")

# ── matched pairs ─────────────────────────────────────────────────────────────────────────────────────────────
CONFIG = ["model_name", "method", "gpu_model", "number_gpus", "number_nodes", "tokens_per_sample", "batch_size",
          "enable_roce", "fast_moe", "fast_kernels", "fms_hf_tuning_version", "torch_dtype", "hidden"]
TOKENS = {  # tokens that encode a factor inside experiment_id; removed so the rest of the protocol matches
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


# ── Fig. 4: throughput vs. batch size on one GPU, and the matched counterpart ─────────────────────────────────
ONE_GPU = {lab: V[(V.gpu == lab) & (V.number_gpus == 1)].groupby("batch_size")[TPS].agg(["median", "size"])
           for lab in ("A100-SXM", "A100-PCIe", "H100-PCIe", "L40S")}
for lab, s in ONE_GPU.items():                 # behind 'throughput grows logarithmically with the batch size'
    x, y = np.log2(s.index.to_numpy()), s["median"].to_numpy()
    slope, r2 = np.polyfit(x, y, 1)[0], np.corrcoef(x, y)[0, 1] ** 2
    put("LogFit" + word(lab), [r2, slope], f"R² {r2:.2f}, {slope:+,.0f} tokens/s per doubling",
        f"least-squares fit of Fig. 4's median throughput on log2(batch size), {lab}, 1 GPU: R², slope")
    one = df[(df.gpu == lab) & (df.number_gpus == 1)]
    steps = {b: speedup(one, "batch_size", b, 2 * b) for b in (1, 2, 4, 8, 16, 32, 64)}
    steps = {b: r for b, r in steps.items() if len(r)}
    put("BatchDoubling" + word(lab), [r.median() for r in steps.values()],
        ", ".join(f"{b}→{2 * b}: {r.median():.2f}×" for b, r in steps.items()),
        f"throughput ratio of matched pairs per batch-size doubling, {lab}, 1 GPU (Fig. 4 compares unmatched "
        "medians, whose mix of LLMs and sequence lengths changes with the batch size)",
        pairs=[len(r) for r in steps.values()])

# ── RoCE, Fast MoE and Fast kernels (Abstract, Sec. 1, Fig. 6) ────────────────────────────────────────────────
multi = df.number_nodes > 1
both = df[multi].groupby("experiment_id").enable_roce.nunique() == 2
C = df[df.experiment_id.isin(both[both].index)]
CAMPAIGN = {"A": C[C.fms_hf_tuning_version == "2.4.0"], "B": C[C.fms_hf_tuning_version == "2.7.1"]}
assert C.experiment_id.nunique() == 4 and len(C) == sum(map(len, CAMPAIGN.values()))

roce = speedup(df[multi], "enable_roce", 0, 1)
roce_by = {c: speedup(D[D.number_nodes > 1], "enable_roce", 0, 1) for c, D in CAMPAIGN.items()}
assert len(roce) == sum(map(len, roce_by.values())), "RoCE pairs outside campaigns A and B"
put("RoceSpeedup", roce.median(), f"{roce.median():.1f}×",
    "RoCE on/off throughput of multi-node runs, median of matched pairs, campaigns A and B pooled", **spread(roce))
by_campaign("RoceSpeedupByCampaign", roce_by, "{:.2f}×", "RoCE on/off throughput of multi-node runs, per campaign")
moe = {e: speedup(df, "fast_moe", 0, e) for e in (1, 2, 4, 8)}
best = max(moe, key=lambda e: moe[e].median())
put("FastMoeMax", moe[best].median(), f"{moe[best].median():.1f}×",
    f"Fast MoE on/off throughput, median of matched pairs, the best expert-parallel degree (EP={best})",
    **spread(moe[best]))
assert set(V.number_gpus[V.fast_moe == 1]) == {1}, "EP=1 ran on more than one GPU"
put("FastMoeByEp", [r.median() for r in moe.values()], ", ".join(f"{r.median():.2f}× (EP={e})" for e, r in moe.items()),
    "Fast MoE on/off throughput per expert-parallel degree (EP=1: one GPU, no expert parallelism)",
    pairs=[len(r) for r in moe.values()])
kernels = speedup(df.assign(fast_kernels=np.where(df.fast_kernels.notna(), "on", "off")), "fast_kernels", "off", "on")
GAIN = {"RoCE": roce, f"Fast MoE, EP={best}": moe[best], "Fast kernels": kernels}
for label, r in GAIN.items():
    g = 100 * (r.median() - 1)
    put("Gain" + word(label.split(",")[0].title()), g, f"+{g:.0f}%",
        f"{label}: median speedup of matched pairs [%] (Fig. 6; RoCE: campaigns A and B pooled)", **spread(r))

# ── weak-scaling efficiency per campaign: within a node (1-8 GPUs) and across nodes (8-32 GPUs) ───────────────


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


for g in (1, 2, 4, 8, 16):
    arms = {"": None} if g < 8 else {"Tcp": False, "Roce": True}
    eff = {net: {} for net in arms}
    for c, D in CAMPAIGN.items():
        e = {net: weak(D, g, on) for net, on in arms.items()}
        same = reduce(pd.Index.intersection, [s.index for s in e.values()])      # configurations every arm ran
        for net in arms:
            eff[net][c] = e[net].loc[same]
    for net, ratios in eff.items():
        where = "within a node" if g < 8 else f"across nodes, {net.upper()}"
        by_campaign(f"WeakEff{g}to{2 * g}{net}", ratios, "{:.2f}",
                    f"weak-scaling efficiency {g}->{2 * g} GPUs ({where}), median of matched pairs, per campaign")

# ── figures (figures.py): file name -> (drawing function, its data) ───────────────────────────────────────────
# A new or restyled figure is one function in figures.py, one entry here, and its page size and plotted-values
# hash in check.py.
FIGURES = {
    "00_failure_rates_by_category": (figures.failures, FAIL),
    "03_performance_vs_batch_size": (figures.batch, ONE_GPU),
    "03_insights_method_scaling": (figures.scaling, {METHOD[k]: V[V.method == k].groupby("number_gpus")[TPS].median()
                                                     for k in METHOD}),
    "07_optimization_roi": (figures.speedups, {lab: (100 * (r.median() - 1), len(r)) for lab, r in GAIN.items()}),
}
FIGS = {name: figures.render(draw, data, OUT / "figures" / f"{name}.pdf") for name, (draw, data) in FIGURES.items()}

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
(OUT / "table1.txt").write_text("\n".join([
    "Table 1. The dataset per GPU type. Ran, Never started, Failed at runtime: share of the experiments.",
    *bordered(TABLE1),
    f"a: GPU types with multi-node jobs | with RoCE | with GPTQ-LoRA: {NUM['OnlySxm']['text']}",
    f"b: GPU types without power data; valid runs with GPU power: {NUM['Table1NoteB']['text']}",
    f"Configuration space: {NUM['NModels']['text']} LLMs ({NUM['ModelParams']['text']}, {NUM['NModelsMoe']['text']} "
    f"MoE); methods Full / LoRA / GPTQ-LoRA {NUM['MethodShares']['text']}% of the experiments; total batch "
    f"{NUM['BatchSizes']['text']}; tokens per sample {NUM['SeqLengths']['text']}; {NUM['NExperimentIds']['text']} "
    f"experiment ids, {NUM['NVersions']['text']} fms-hf-tuning versions; {NUM['NRunsRoce']['text']} runs with RoCE."])
    + "\n", encoding="utf-8")
print(f"{SPLIT}: {DATASET['rows']:,} rows x {DATASET['columns']} columns from {', '.join(DATASET['files'])}")
print(f"wrote {len(NUM)} numbers and the data of {len(FIGS)} figures to out/numbers.json, Table 1 to "
      f"out/table1.txt, the figures to out/figures/")
