"""Regenerate the paper's five figures (Figs. 2-6) from IBM's public LLMFineTuningBench dataset into out/, one PDF per
figure, named after its number in the paper (fig2_*.pdf to fig6_*.pdf). bash reproduce.sh installs the pinned packages
and runs this script.

Data: the dataset on the Hugging Face Hub (ibm-research/LLMFineTuningBench, Apache-2.0, IBM Research), loaded as its
page shows. If that fails (no network, the dataset moved) or the Hub no longer serves the table the paper analyzed,
the snapshot of its file in data/ado-sfttrainer.csv. The script prints which of the two it used.

The paper's figures were drawn with matplotlib 3.11.0 in Times New Roman (regular and bold, embedded as TrueType);
with both, this script draws them identically. matplotlib 3.10 places the text up to 1.7 pt elsewhere, and without
Times New Roman matplotlib would silently draw in another font, so render() refuses to draw without it.

Definitions
- Outcome (Fig. 2). Valid: is_valid == 1. An invalid experiment was rejected before launch (never started) if its
  configuration breaks a validation rule of the actuator: the number of GPUs does not divide the total batch size,
  the expert-parallel degree does not divide the number of GPUs, or the number of nodes does not divide the number
  of GPUs. Every other invalid experiment failed at runtime. The failure ratio counts both.
- Throughput (Figs. 3-5): dataset_tokens_per_second of a valid run (all GPUs of the job together).
- Matched pair (Fig. 6): the valid runs of one configuration with and without an optimization. All other
  configuration columns, the settings recorded only in the identifier (`hidden`) and the protocol (experiment_id
  without the optimization's own token) are equal. A pair's speedup divides the median throughputs of its two arms;
  a bar is the median over pairs, its whisker the interquartile range.
- RoCE (Fig. 6a): the only two campaigns that ran multi-node jobs both with and without RoCE disagree, so they are
  shown apart. fms-hf-tuning 2.4.0 runs without a time limit; 2.7.1 stops every run after stop_after_seconds (600 s).
"""
import re
import sys
import warnings
from contextlib import contextmanager
from pathlib import Path

import matplotlib
matplotlib.use("Agg")                         # other backends write narrower PDFs than declared
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
SNAPSHOT, OUT = HERE / "data" / "ado-sfttrainer.csv", HERE / "out"
if matplotlib.__version__ != "3.11.0":
    print(f"warning: matplotlib {matplotlib.__version__}; the paper's figures were drawn with 3.11.0, which this "
          "script needs to draw them identically (bash reproduce.sh installs it)", file=sys.stderr)

# ── data: the public dataset, loaded as its Hub page shows (Use this dataset); else the snapshot ─────────────────────
warnings.filterwarnings("ignore", message="The 'verbose' keyword in pd.read_csv is deprecated")  # datasets 2.13 uses it
snapshot = pd.read_csv(SNAPSHOT)
try:
    import datasets
    datasets.disable_progress_bar()          # its row counter is throttled (it stops at 30,000); rows are counted below
    datasets.logging.set_verbosity_error()
    from datasets import load_dataset
    ds = load_dataset("ibm-research/LLMFineTuningBench")
    (split,) = ds.values()                   # the dataset's only split
    df = split.to_pandas()
    pd.testing.assert_frame_equal(df, snapshot, check_dtype=False, check_exact=True)   # the table the paper analyzed
    source = "the Hugging Face Hub (ibm-research/LLMFineTuningBench)"
except Exception as error:                   # no network, the dataset moved or changed
    df = snapshot
    why = ("its table differs from the snapshot" if isinstance(error, AssertionError)
           else f"{type(error).__name__}: {' '.join(str(error).split())[:200]}")
    source = f"the snapshot data/ado-sfttrainer.csv (the Hugging Face Hub failed: {why})"
print(f"data: {len(df):,} rows x {df.shape[1]} columns from {source}")

TPS = "dataset_tokens_per_second"
GPU = {"NVIDIA-A100-SXM4-80GB": "A100-SXM", "NVIDIA-A100-80GB-PCIe": "A100-PCIe", "L40S": "L40S",
       "NVIDIA-H100-PCIe": "H100-PCIe"}                                    # label per GPU type, in Table 1 order
METHOD = {"full": "Full", "lora": "LoRA", "gptq-lora": "GPTQ-LoRA"}

# ── derived columns ─────────────────────────────────────────────────────────────────────────────────────────────────
assert (df.number_gpus >= 1).all() and set(df.gpu_model) == set(GPU)
df["gpu"] = df.gpu_model.map(GPU)
ep = df.fast_moe.fillna(0)
rejected = ((df.batch_size % df.number_gpus != 0)                          # the GPUs do not divide the total batch
            | ((ep > 0) & (df.number_gpus % ep.where(ep > 0, 1) != 0))      # the EP degree does not divide the GPUs
            | (df.number_gpus % df.number_nodes != 0))                      # the nodes do not divide the GPUs
df["outcome"] = np.select([df.is_valid == 1, rejected], ["valid", "rejected"], "runtime")

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


# ── Fig. 2: failure ratio per batch size, sequence length, GPU type and fine-tuning method ─────────────────────────
def failure_table(key):
    """Experiments, valid ones, failing ones (rejected + runtime) and rejected ones per value of key."""
    c = pd.crosstab(key, df.outcome).reindex(columns=["valid", "rejected", "runtime"], fill_value=0)
    return pd.DataFrame({"runs": c.sum(axis=1), "valid": c.valid, "failing": c.rejected + c.runtime,
                         "rejected": c.rejected})


FAIL = {"batch": failure_table(df.batch_size.astype(int)), "seq": failure_table(df.tokens_per_sample.astype(int)),
        "gpu": failure_table(df.gpu), "method": failure_table(df.method.map(METHOD))}
# per panel, bottom to top (batch sizes in order, the other panels by increasing failure ratio)
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

# ── Fig. 3: median throughput per batch size, one GPU, per GPU type ──────────────────────────────────────────────────
ONE = V[V.number_gpus == 1]
FIG3 = {}
for g, lab in GPU.items():
    tps = ONE[ONE.gpu_model == g].groupby("batch_size")[TPS].median()
    FIG3[lab] = {"batch_size": tps.index.astype(int).tolist(), "median_tps": tps.tolist()}

# ── Fig. 4: median throughput per batch size and sequence length ────────────────────────────────────────────────────
BATCHES, SEQS = (sorted(int(v) for v in df[c].unique()) for c in ("batch_size", "tokens_per_sample"))
cell = lambda d: [d.batch_size.astype(int), d.tokens_per_sample.astype(int)]
grid = lambda s: s.unstack().reindex(index=BATCHES, columns=SEQS)
med = grid(V.groupby(cell(V))[TPS].median())
n_all = grid(df.groupby(cell(df)).size()).fillna(0)
FIG4 = {"batch_size": BATCHES, "sequence_length": SEQS,
        "median_tps": [[None if np.isnan(x) else float(x) for x in row] for row in med.to_numpy(float)],
        "experiments": n_all.astype(int).to_numpy().tolist()}

# ── Fig. 5: median throughput per number of GPUs, per fine-tuning method ─────────────────────────────────────────────
FIG5 = {}
for m, label in METHOD.items():
    tps = V[V.method == m].groupby("number_gpus")[TPS].median()
    FIG5[label] = {"number_gpus": tps.index.astype(int).tolist(), "median_tps": tps.tolist()}

# ── Fig. 6: speedup of RoCE, Fast MoE and Fast Kernels over matched pairs ───────────────────────────────────────────
CONFIG = ["model_name", "method", "gpu_model", "number_gpus", "number_nodes", "tokens_per_sample", "batch_size",
          "enable_roce", "fast_moe", "fast_kernels", "fms_hf_tuning_version", "torch_dtype", "hidden"]
TOKENS = {  # tokens that encode an optimization inside experiment_id; removed so the rest of the protocol matches
    "enable_roce": [r"-enable_roce\.[^-]+"],
    "fast_kernels": [r"-fast_kernels\.\[[^\]]*\]"],
}


def protocol(exp_id, factor):
    for pattern in TOKENS.get(factor, []):
        exp_id = re.sub(pattern, "", exp_id)
    return exp_id


def speedup(d, factor, a, b):
    """Median throughput at factor == b over that at factor == a, one ratio per matched pair."""
    d = d[d.outcome == "valid"].copy()
    for c in ("fast_kernels", "fms_hf_tuning_version", "torch_dtype"):
        d[c] = d[c].fillna("none").astype(str)
    d[["fast_moe", "enable_roce"]] = d[["fast_moe", "enable_roce"]].fillna(0)
    d["protocol"] = d.experiment_id.map(lambda e: protocol(e, factor))
    keys = [c for c in CONFIG if c != factor] + ["protocol"]
    arm = lambda v: d[d[factor] == v].groupby(keys, dropna=False)[TPS].median()
    j = pd.concat([arm(a).rename("a"), arm(b).rename("b")], axis=1, join="inner")
    return j.b / j.a


def level(r, key):
    """The value of one configuration column for each pair of r."""
    return r.index.get_level_values(key)


multi = df.number_nodes > 1
both = df[multi].groupby("experiment_id").enable_roce.nunique() == 2
C = df[df.experiment_id.isin(both[both].index)]                    # the two RoCE campaigns, by fms-hf-tuning version
LIMIT = {v: C.experiment_id[C.fms_hf_tuning_version == v].str.extract(r"-stop_after_seconds\.([\d.]+)",
                                                                       expand=False).astype(float)
         for v in ("2.4.0", "2.7.1")}                              # [s] the time limit in the experiment_id, if any
assert LIMIT["2.4.0"].isna().all() and LIMIT["2.7.1"].notna().all() and LIMIT["2.7.1"].nunique() == 1
CAP = LIMIT["2.7.1"].iloc[0]
roce = speedup(df[multi], "enable_roce", 0, 1)
version = level(roce, "fms_hf_tuning_version")
assert set(version) == set(LIMIT), "RoCE pairs outside the two campaigns"
ra, rb = roce[version == "2.4.0"], roce[version == "2.7.1"]

EP = sorted(int(e) for e in V.fast_moe.dropna().unique() if e > 0)   # expert-parallel degrees with valid runs
moe = {e: speedup(df, "fast_moe", 0, e) for e in EP}
by_ep = lambda m: {e: r[level(r, "method") == m] for e, r in moe.items()}
moe_full, moe_lora = by_ep("full"), by_ep("lora")

kernels = speedup(df.assign(fast_kernels=np.where(df.fast_kernels.notna(), "on", "off")), "fast_kernels", "off", "on")


def fig6_bar(group, label, r):
    """One bar of Fig. 6: its matched ratios' count and quartiles."""
    return {"group": group, "label": label, "n": len(r), "p25": float(r.quantile(0.25)), "p50": float(r.median()),
            "p75": float(r.quantile(0.75))}


FIG6 = [fig6_bar("RoCE", "no time limit", ra), fig6_bar("RoCE", f"{CAP:g} s limit", rb),
        *(fig6_bar("Fast MoE", f"{m}, EP {e}", r[e]) for m, r in (("Full", moe_full), ("LoRA", moe_lora)) for e in EP),
        fig6_bar("Fast Kernels", "all pairs", kernels)]

# ═════ drawing: one function per figure; the data computed above go in, the figure comes out ══════════════════════

FONT = "Times New Roman"
FONT_FILES = {"normal": "TimesNewRomanPSMT", "bold": "TimesNewRomanPS-BoldMT"}  # PostScript names in the paper's PDFs

# ── geometry ─────────────────────────────────────────────────────────────────────────────────────────────────────
# The paper's page (acmart, sigplan): \textwidth 505.89 pt and \columnsep 24 pt [TeX pt]. Fig. 2 is drawn 7.0 in wide,
# cropped to its content and printed at \textwidth; Figs. 3-6 are drawn at \columnwidth and printed at scale 1.
TEXT_W = 505.89 / 72.27                      # [in] 7.000
COL_W = (505.89 - 24.0) / 2 / 72.27          # [in] 3.334: a page of 240.04 bp
# The column figures' sizes are Fig. 2's times 1.048: \textwidth over the 480.852 bp that Fig. 2 was wide when their
# sizes were matched to it. The paper's PDFs were drawn with this factor, so it stays fixed here.
SCALE = TEXT_W / (480.852 / 72)


def s(pt):
    """A Fig. 2 size [pt] as the column figures print it (rounded to 0.01 pt, as the paper's PDFs were drawn)."""
    return round(pt * SCALE, 2)


# ── style ────────────────────────────────────────────────────────────────────────────────────────────────────────
# The paper's figure style, written out as its differences from matplotlib's defaults (so seaborn is not needed):
# seaborn's "white" style, then Times New Roman, a light solid grid, square legend frames and TrueType fonts in PDFs.
# The font is named, not given as the "serif" alias, so that it cannot resolve to another font when the figure is saved.
PAPER = {
    "axes.edgecolor": ".15", "axes.labelcolor": ".15", "text.color": ".15", "xtick.color": ".15", "ytick.color": ".15",
    "lines.solid_capstyle": "round", "patch.edgecolor": "w", "patch.force_edgecolor": True,
    "font.family": [FONT], "font.serif": [FONT, "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "axes.grid": True, "axes.axisbelow": True, "grid.color": "#d9d9d9",
    "legend.facecolor": "white", "legend.edgecolor": "black", "legend.fancybox": False, "legend.framealpha": 1.0,
    "legend.borderpad": 0.3,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
}
FULL_WIDTH = {**PAPER, "font.size": 9.0, "axes.titlesize": 8.0, "axes.labelsize": 9.0, "xtick.labelsize": 8.5,
              "ytick.labelsize": 8.5, "legend.fontsize": 9.0, "hatch.linewidth": 0.6}                  # Fig. 2
COLUMN = {**PAPER, "font.size": s(9.0), "axes.labelsize": s(9.0), "axes.labelweight": "bold", "axes.titlesize": s(8.0),
          "xtick.labelsize": s(8.5), "ytick.labelsize": s(8.5), "legend.fontsize": s(8.0), "grid.linewidth": s(0.8),
          "axes.linewidth": s(1.25), "xtick.major.width": s(1.25), "ytick.major.width": s(1.25),
          "xtick.major.size": s(3.0), "ytick.major.size": s(3.0)}                                      # Figs. 3-6
RED = "#CC3311"                                                          # Fig. 2's failures
BLUE, ORANGE, CYAN, MAGENTA = "#0077BB", "#EE7733", "#33BBEE", "#EE3377"  # a colour-blind-safe palette ("vibrant")
COUNT_GREY = "0.3"                                                       # counts at the right of a panel
comma = mticker.FuncFormatter(lambda v, _p: f"{v:,.0f}")


class _Figure(Figure):
    """A figure that is saved in the style it was drawn in, whatever the caller's rcParams: the fonts, the ticks that
    axes add lazily and the PDF font type are all resolved when the file is written."""
    rc, save_kw = PAPER, {}

    def savefig(self, *args, **kwargs):
        with mpl.style.context(["default", self.rc]):
            return super().savefig(*args, **{**self.save_kw, **kwargs})


@contextmanager
def styled(rc, **save_kw):
    """Draw in matplotlib's defaults plus rc; figures created here save in the same style (see _Figure)."""
    with mpl.style.context(["default", rc]):
        yield type("Styled", (_Figure,), {"rc": rc, "save_kw": save_kw})


def require_fonts():
    """Raise unless Times New Roman resolves to the paper's two fonts: matplotlib would only warn and draw in another
    font."""
    for weight, ps_name in FONT_FILES.items():
        try:
            path = font_manager.findfont(font_manager.FontProperties(family=FONT, weight=weight),
                                         fallback_to_default=False)
            found = font_manager.get_font(path).postscript_name
        except ValueError:
            found = None
        if found != ps_name:
            raise RuntimeError(f"the paper's figures need {FONT} ({weight}, {ps_name}); found {found or 'none'}. "
                               "macOS ships it; on Debian or Ubuntu install ttf-mscorefonts-installer (contrib, "
                               "multiverse), then delete matplotlib's font cache (.venv/matplotlib, as reproduce.sh "
                               "sets it)")


def render(draw, data, pdf):
    """Draw one figure into pdf."""
    require_fonts()
    fig = draw(data)
    fig.savefig(pdf, metadata={"CreationDate": None})          # no date: the same data give the same file
    plt.close(fig)


def square_legend(leg):
    """A thin, square, black, white-filled legend frame."""
    frame = leg.get_frame()
    frame.set_edgecolor("black")
    frame.set_linewidth(0.8)
    frame.set_facecolor("white")
    frame.set_alpha(1.0)
    frame.set_boxstyle("square", pad=0)


def bar_panel(ax, ticks, spine_w, tick_len, tick_w, row_w):
    """A panel of horizontal bars: black left and bottom spines, inner ticks on the right (one per bar), faint lines
    between the bars."""
    for yy in range(len(ticks) - 1):
        ax.axhline(yy + 0.5, color="0.85", linewidth=row_w, zorder=0.5)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_color("black")
        ax.spines[side].set_linewidth(spine_w)
    sec = ax.secondary_yaxis("right")
    sec.set_yticks(ticks)
    sec.set_yticklabels([])
    sec.tick_params(direction="in", length=tick_len, width=tick_w)


def count(ax, x, y, n, size, zorder):
    """The number n in grey, right-aligned at x; its white box hides the grid line behind it."""
    ax.text(x, y, f"{n:,}", ha="right", va="center", fontsize=size, color=COUNT_GREY, zorder=zorder,
            bbox=dict(facecolor="white", edgecolor="none", boxstyle="square,pad=0.08"))


# ── Fig. 2: failure ratio per batch size, sequence length, GPU model and fine-tuning method ───────────────────────────
FAIL_XMAX = 132                              # [%] room for the counts; the ticks stay at 0, 50 and 100
FAIL_TITLES = {"batch": "(a) Batch Sizes", "seq": "(b) Sequence Lengths", "gpu": "(c) GPU Models",
               "method": "(d) Fine-tuning Methods"}


def _failure_panel(ax, d, title):
    runs, failing, rejected = (np.array(d[k]) for k in ("runs", "failing", "rejected"))
    rate = (1 - (runs - failing) / runs) * 100   # failure ratio [%]
    rej = 100 * (rejected / runs)                # rejected before launch [%]
    run = rate - rej                             # failed at runtime [%]
    ax.barh(d["label"], rej, color=RED, edgecolor="white", linewidth=0, hatch="////")
    ax.barh(d["label"], run, left=rej, color=RED, edgecolor="none", linewidth=0)
    bars = ax.barh(d["label"], rate, fill=False, edgecolor="black", linewidth=0.8)    # outline of the whole bar
    for bar, rj, r in zip(bars, rej, rate):                                          # divider between the parts
        if 0 < rj < r:
            ax.plot([rj, rj], [bar.get_y(), bar.get_y() + bar.get_height()], color="black", lw=0.6, zorder=3)
    for bar, r, n in zip(bars, rate, failing):
        y = bar.get_y() + bar.get_height() / 2
        # the ratio in white at the end of its bar, with a thin red halo that keeps it readable over the hatching
        ax.text(r - 2.5, y, f"{r:.0f}%", ha="right", va="center", color="white", fontsize=7, fontweight="bold",
                zorder=4, path_effects=[pe.withStroke(linewidth=1.8, foreground=RED)])
        count(ax, FAIL_XMAX - 6, y, int(n), 6.5, zorder=3)
    ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
    ax.set_xlim(0, FAIL_XMAX)
    ax.set_xticks([0, 50, 100])
    ax.grid(True, axis="x")
    ax.grid(False, axis="y")
    ax.set_axisbelow(True)
    ax.axvline(100, color="0.55", linewidth=0.8, zorder=0.9)      # the 100% line, darker than the grid
    ax.tick_params(axis="x", which="both", bottom=True, length=3)
    bar_panel(ax, ax.get_yticks(), spine_w=0.8, tick_len=2.5, tick_w=0.6, row_w=0.5)


def failures(data):
    """Fig. 2 (7.0 x 1.85 in, cropped to its content with a 0.02 in margin: 457.86 x 129.01 pt). data: per panel
    (batch, seq, gpu, method), bottom to top, the bar labels and the experiments, failing experiments and experiments
    rejected before launch. Each bar splits into the rejected (hatched) and runtime (solid) shares."""
    with styled(FULL_WIDTH, bbox_inches="tight", pad_inches=0.02) as fig_class:
        fig, axes = plt.subplots(1, 4, figsize=(7.0, 1.85), FigureClass=fig_class)
        for ax, (key, title) in zip(axes, FAIL_TITLES.items()):
            _failure_panel(ax, data[key], title)
            if key == "batch":                                   # eleven bars: smaller type
                ax.tick_params(axis="y", labelsize=6.5)
                for t in ax.texts:
                    t.set_fontsize(5.5)
        fig.subplots_adjust(left=0.1, right=0.95, top=0.88, bottom=0.24, wspace=0.42)
        fig.supxlabel("Failure Ratio [%]", x=(0.1 + 0.95) / 2, y=0.03, fontweight="bold", fontsize=9)
        fig.legend(handles=[Patch(facecolor=RED, edgecolor="white", hatch="////", label="rejected before launch"),
                            Patch(facecolor=RED, edgecolor="black", label="failed at runtime")],
                   loc="lower right", bbox_to_anchor=(0.97, -0.03), ncol=2, frameon=False, fontsize=7,
                   handlelength=1.6)
    return fig


# ── Figs. 3 and 5: median throughput per batch size (one GPU) and per number of GPUs ─────────────────────────────────
LINE_HEIGHT = 1.40                                                # [in] (100.8 bp)
MSIZE = {"o": 4.6, "s": 4.1, "D": 3.7, "^": 4.8}                  # about equal marker areas [pt]


def _line_figure(fig_class):
    fig, ax = plt.subplots(figsize=(COL_W, LINE_HEIGHT), FigureClass=fig_class)
    w, h = fig.get_size_inches()                  # margins [in]: label bands independent of the height
    fig.subplots_adjust(left=0.82 / w, right=1 - 0.085 / w, top=1 - 0.255 / h, bottom=0.39 / h)
    return fig, ax


def _lines(ax, data, x, colors, markers, xlim, ylim):
    """One line per series (label -> x values under x, medians under 'median_tps'); every point inside the axes."""
    assert list(data) == list(colors), list(data)
    assert all(xlim[0] < v <= xlim[1] for d in data.values() for v in d[x]), "a point outside the x axis"
    assert all(ylim[0] <= v <= ylim[1] for d in data.values() for v in d["median_tps"]), "a point outside the y axis"
    for label, d in data.items():
        m = markers[label]
        ax.plot(d[x], d["median_tps"], color=colors[label], lw=s(1.2), marker=m, ms=MSIZE[m], mec="black",
                mew=s(0.5), label=label, zorder=3, clip_on=False)


def _line_axes(fig, ax, xlim, ylim, xticks, xlabels, yticks, xlabel, ncol):
    """Fixed log2 x axis, y axis with thousands separators, and a framed one-row legend above the plot."""
    ax.set_xscale("log", base=2)
    ax.set_xticks(xticks, xlabels)
    ax.xaxis.set_minor_locator(mticker.NullLocator())
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(comma)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Median Throughput\n[tokens/s]")
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", which="major", length=s(3))
    hs, ls = ax.get_legend_handles_labels()
    square_legend(fig.legend(hs, ls, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=ncol, handlelength=1.5,
                             handletextpad=0.35, columnspacing=0.8, borderaxespad=0.0, fontsize=s(7.0)))


def batch(data):
    """Fig. 3 (240.04 x 100.8 pt): median throughput of the valid one-GPU runs per total batch size, one line per GPU
    type. data: GPU label -> batch sizes and median throughputs [tokens/s], in Table 1 order."""
    xlim, ylim, ticks = (0.8, 128 * 1.25), (0, 10_000), [2 ** k for k in range(8)]      # batch sizes 1 .. 128
    with styled(COLUMN) as fig_class:
        fig, ax = _line_figure(fig_class)
        _lines(ax, data, "batch_size", {"A100-SXM": BLUE, "A100-PCIe": ORANGE, "L40S": CYAN, "H100-PCIe": MAGENTA},
               {"A100-SXM": "o", "A100-PCIe": "s", "L40S": "D", "H100-PCIe": "^"}, xlim, ylim)
        _line_axes(fig, ax, xlim, ylim, ticks, [f"{t:,}" for t in ticks], [0, 2_500, 5_000, 7_500, 10_000],
                   "Batch Size [#]", ncol=4)
    return fig


def scaling(data):
    """Fig. 5 (240.04 x 100.8 pt): median throughput of the valid runs per number of GPUs, one line per fine-tuning
    method (not matched). The dashed line marks the node boundary (8 GPUs per node). data: method label -> GPU
    counts and median throughputs [tokens/s], in legend order."""
    xlim, ylim, ticks = (0.8, 40), (0, 36_000), [1, 2, 4, 8, 16, 32]
    boundary = 8 * 2 ** 0.12                   # just right of 8 GPUs (1 node); 16 GPUs need 2 nodes
    with styled(COLUMN) as fig_class:
        fig, ax = _line_figure(fig_class)
        ax.axvline(boundary, color="0.25", lw=s(0.8), ls=(0, (3, 2)), zorder=2)
        _lines(ax, data, "number_gpus", {"Full": BLUE, "LoRA": ORANGE, "GPTQ-LoRA": CYAN},
               {"Full": "o", "LoRA": "s", "GPTQ-LoRA": "^"}, xlim, ylim)
        kw = dict(fontsize=s(6.5), color="0.3", va="bottom")
        ax.text(boundary / 1.12, 1_300, "1 node", ha="right", **kw)
        ax.text(boundary * 1.12, 1_300, "2–4 nodes", ha="left", **kw)
        _line_axes(fig, ax, xlim, ylim, ticks, [str(t) for t in ticks], [0, 10_000, 20_000, 30_000],
                   "Number of GPUs [#]", ncol=3)
    return fig


# ── Fig. 4: median throughput per batch size and sequence length ──────────────────────────────────────────────────────
HEAT_BOUNDS = [0, 5_000, 15_000, 30_000, 50_000, 80_000]          # colour buckets [tokens/s]
HEAT_DPI = 300                  # the heatmap is an image sampled at the figure's resolution (555 x 359 px)
EMPTY_GREY = "0.6"              # outline of a cell without a valid run, hatching of a cell without experiments


def compact(v):
    """A cell label: 1.4k, 75k (one decimal below ten thousand, none above)."""
    x = v / 1000
    t = f"{x:.1f}" if x < 10 else f"{x:.0f}"
    return t.removesuffix(".0") + "k"


def heatmap(data):
    """Fig. 4 (240.04 x 115.2 pt). data: batch sizes (rows), sequence lengths (columns), and per cell the median
    throughput of the valid runs [tokens/s] (None: no valid run) and the experiments. A cell with experiments but no
    valid run is blank with a grey outline; a cell without experiments is hatched."""
    batch_sizes, seq = data["batch_size"], data["sequence_length"]
    m = np.array([[np.nan if v is None else v for v in row] for row in data["median_tps"]], dtype=float)
    nall = data["experiments"]
    cmap = ListedColormap([mpl.colormaps["Blues"](x) for x in np.linspace(0.08, 1.0, len(HEAT_BOUNDS) - 1)])
    norm = BoundaryNorm(HEAT_BOUNDS, ncolors=cmap.N)
    fs_small, fs_inbar = s(6.5), s(7.0)           # batch-size ticks, cell labels, key | the other ticks, colour bar
    with styled(COLUMN) as fig_class:
        W, H = COL_W, 1.60
        fig = plt.figure(figsize=(W, H), dpi=HEAT_DPI, FigureClass=fig_class)
        b_in, t_in = 0.385, 0.02                      # x-axis band, top margin [in]
        ax_h = H - b_in - t_in
        ax = fig.add_axes([0.175, b_in / H, 0.555, ax_h / H])
        cb_h = 0.62 * ax_h                            # colour bar: upper 62% of the plot height
        cax = fig.add_axes([0.755, (H - 0.06 - cb_h) / H, 0.03, cb_h / H])
        im = ax.imshow(np.ma.masked_invalid(m), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
        nr, nc = m.shape
        dark = len(HEAT_BOUNDS) - 3                   # the two darkest buckets carry white text
        dx, dy = 0.03, 0.07                           # inset the outline inside the white separators
        for i in range(nr):
            for j in range(nc):
                if nall[i][j] == 0:                   # no experiment: hatched
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="white", edgecolor=EMPTY_GREY,
                                           hatch="//////", lw=0, zorder=1))
                elif np.isnan(m[i, j]):               # experiments, no valid run: blank, thin grey outline
                    ax.add_patch(Rectangle((j - 0.5 + dx, i - 0.5 + dy), 1 - 2 * dx, 1 - 2 * dy, facecolor="white",
                                           edgecolor=EMPTY_GREY, lw=s(0.6), zorder=2.6))
                else:
                    b = int(np.digitize(m[i, j], HEAT_BOUNDS[1:-1]))
                    ax.text(j, i, compact(m[i, j]), ha="center", va="center", fontsize=fs_small,
                            color="white" if b >= dark else "black", zorder=3)
        # white separators between cells, drawn above the hatching
        ax.set_xticks(np.arange(-0.5, nc, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, nr, 1), minor=True)
        ax.grid(False)
        ax.grid(which="minor", color="white", linewidth=s(0.8))
        ax.tick_params(which="minor", length=0)
        ax.set_axisbelow(False)
        ax.set_xticks(range(nc), [f"{q:,}" for q in seq])
        ax.set_yticks(range(nr), [f"{q:,}" for q in batch_sizes])
        ax.tick_params(axis="y", labelsize=fs_small, length=s(2.5))
        ax.tick_params(axis="x", labelsize=fs_inbar, length=s(2.5))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_xlabel("Sequence Length [tokens]", fontsize=s(9.0))
        ax.set_ylabel("Batch Size [#]", fontsize=s(9.0))
        cb = fig.colorbar(im, cax=cax, ticks=HEAT_BOUNDS, spacing="uniform")
        cb.ax.set_yticklabels(["0"] + [f"{q // 1000}k" for q in HEAT_BOUNDS[1:]], fontsize=fs_inbar)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=s(2.5))
        fig.text(0.94, (H - 0.06 - cb_h / 2) / H, "Median\nThroughput\n[tokens/s]", rotation=90, ha="center",
                 va="center", fontsize=fs_inbar, fontweight="bold")
        # key for the two kinds of empty cell
        handles = [Patch(facecolor="white", edgecolor=EMPTY_GREY, lw=s(0.6), label="no valid run"),
                   Patch(facecolor="white", edgecolor=EMPTY_GREY, hatch="//////", lw=s(0.6), label="no experiment")]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.745, (b_in - 0.03) / H), ncol=1,
                   handlelength=1.1, handleheight=1.0, handletextpad=0.4, borderpad=0.35,
                   labelspacing=0.35, fontsize=fs_small, frameon=False)
    return fig


# ── Fig. 6: speedup of RoCE, Fast MoE and Fast Kernels over matched pairs ─────────────────────────────────────────────
SPEED_XMAX = 4.1                                  # past the longest whisker (3.51) and the counts


def speedups(rows):
    """Fig. 6 (240.04 x 183.6 pt): three stacked panels on one speedup axis. rows, top to bottom: panel (group), row
    label, pairs (n) and the quartiles of the pairs' speedups (p25, p50 = bar, p75 = whisker ends)."""
    groups = list(dict.fromkeys(r["group"] for r in rows))
    per = {g: [r for r in rows if r["group"] == g] for g in groups}
    with styled(COLUMN) as fig_class:
        fig, axes = plt.subplots(len(groups), 1, figsize=(COL_W, 2.55), sharex=True, FigureClass=fig_class,
                                 gridspec_kw={"height_ratios": [len(per[g]) for g in groups]})
        w, h = fig.get_size_inches()                                # margins [in]
        fig.subplots_adjust(left=0.93 / w, right=1 - 0.05 / w, top=1 - 0.19 / h, bottom=0.42 / h)
        fig.subplots_adjust(hspace=0.48)
        for i, (ax, g) in enumerate(zip(axes, groups)):
            rs = per[g]
            y = np.arange(len(rs))
            p50 = np.array([r["p50"] for r in rs])
            p25 = np.array([r["p25"] for r in rs])
            p75 = np.array([r["p75"] for r in rs])
            ax.barh(y, p50, height=0.76, color=BLUE, edgecolor="black", linewidth=s(0.8), zorder=3)
            lw = s(1.0)                                             # interquartile range: whisker with caps
            ax.hlines(y, p25, p75, color=ORANGE, lw=lw, zorder=4)
            ax.vlines(p25, y - 0.2, y + 0.2, color=ORANGE, lw=lw, zorder=4)
            ax.vlines(p75, y - 0.2, y + 0.2, color=ORANGE, lw=lw, zorder=4)
            for yy, m, hi, r in zip(y, p50, p75, rs):
                x_out = max(m, hi) + 0.1
                if x_out + 0.52 < SPEED_XMAX - 0.47:                # room right of the whisker: black value there
                    ax.text(x_out, yy, f"{m:.2f}×", ha="left", va="center", color="black",
                            fontsize=s(8.0), fontweight="bold", zorder=5)
                else:                                               # long bar: white value at its left end
                    ax.text(0.08, yy, f"{m:.2f}×", ha="left", va="center", color="white",
                            fontsize=s(8.0), fontweight="bold", zorder=5)
                count(ax, SPEED_XMAX - 0.16, yy, r["n"], s(7.2), zorder=5)
            for yy, m in zip(y, p50):                               # 1x: white over a bar, black past its end
                ax.vlines(1.0, yy - 0.38, yy + 0.38, color="white" if m >= 1 else "black",
                          lw=s(0.8), linestyles=[(0, (3, 2))], zorder=3.6)
            ax.set_yticks(y, [r["label"] for r in rs], fontsize=s(8.5))
            ax.set_ylim(len(rs) - 0.5, -0.5)
            ax.set_title(f"({'abc'[i]}) {g}", loc="left", fontsize=s(9.0), fontweight="bold", pad=s(2.5))
            ax.grid(True, axis="x")
            ax.grid(False, axis="y")
            ax.set_axisbelow(True)
            bar_panel(ax, y, spine_w=s(0.8), tick_len=s(2.5), tick_w=s(0.6), row_w=s(0.5))
            ax.tick_params(axis="y", length=s(3))
            ax.tick_params(axis="x", which="both", bottom=True, length=s(3))
        axes[-1].set_xlim(0, SPEED_XMAX)
        axes[-1].set_xticks([0, 1, 2, 3], ["0", "1", "2", "3"])
        axes[-1].set_xlabel("Speedup [×]")
    return fig


# ── write the figures, named after their number in the paper (Figs. 2-6) ──────────────────────────────────────────────
FIGURES = {
    "fig2_failure_ratios": (failures, FIG2),
    "fig3_throughput_vs_batch_size": (batch, FIG3),
    "fig4_throughput_heatmap": (heatmap, FIG4),
    "fig5_gpu_scaling": (scaling, FIG5),
    "fig6_optimization_speedups": (speedups, FIG6),
}
OUT.mkdir(exist_ok=True)
for name, (draw, data) in FIGURES.items():
    render(draw, data, OUT / f"{name}.pdf")
print(f"wrote {len(FIGURES)} figures to out/: " + ", ".join(f"{name}.pdf" for name in FIGURES))
