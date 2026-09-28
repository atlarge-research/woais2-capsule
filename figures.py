"""The paper's figures, one function each: the data reproduce.py computed go in; the figure and its plotted values
come out. reproduce.py lists them in FIGURES; check.py pins each one's page size and plotted values."""
import hashlib

import matplotlib
matplotlib.use("Agg")                         # other backends write narrower PDFs than declared
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.axisbelow": True, "grid.color": "#D9D9D9", "grid.linewidth": 0.8,
                     "legend.fancybox": False, "legend.edgecolor": "black", "pdf.fonttype": 42})
BLUE, VERMILION, GREEN, ORANGE, RED = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#CC3311"
# [in] The page of the draft's single-column PDFs: 0.48 of a 441.02 pt text width, which is not the paper's. The
# paper scales it to its 240.9 pt column (x1.14), and Fig. 3's page to its 505.9 pt text width (x1.05).
COLUMN = (0.48 * 441.01773 / 72.27, 2.0)
comma = mticker.FuncFormatter(lambda v, _: f"{v:,.0f}")


def render(draw, data, pdf):
    """Draw one figure into pdf; return the file's SHA-256 and the plotted values."""
    fig, plotted = draw(data)
    fig.savefig(pdf, metadata={"CreationDate": None})
    plt.close(fig)
    return {"pdf_sha256": hashlib.sha256(pdf.read_bytes()).hexdigest(), "data": plotted}


def log2_axis(ax, ticks):
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(comma)
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.yaxis.set_major_formatter(comma)
    ax.grid(axis="y")


def failures(fail):
    """Fig. 3: failure ratio per GPU type, method, batch size and sequence length (panel: table, bottom to top)."""
    # The draft's layout (7.0 x 1.85 in; subplots_adjust(left=.1, right=.95, bottom=.24, top=.88, wspace=.75))
    # cropped to its tight bounding box, in points: page 480.85 x 122.82, panels 68.544 x 85.248, 119.952 apart.
    w, h = 480.85175, 122.8155
    fig = plt.figure(figsize=(w / 72, h / 72))
    titles = ["(a) GPU Models", "(b) Fine-tuning Methods", "(c) Batch Sizes", "(d) Sequence Lengths"]
    for k, ((panel, t), title) in enumerate(zip(fail.items(), titles)):
        ax = fig.add_axes(((48.59625 + 119.952 * k) / w, 27.705625 / h, 68.544 / w, 85.248 / h))
        small = panel == "batch"                                     # eleven bars: smaller type
        ax.barh(t.index, t.failure_pct, color=RED, edgecolor="black", linewidth=0.8)
        for y, (r, n) in enumerate(zip(t.failure_pct, t.failing)):   # ratio at the bar end, count at the right
            ax.text(r - 2.5, y, f"{r:.0f}%", ha="right", va="center", color="white", fontweight="bold",
                    fontsize=5.5 if small else 7)
            ax.text(130, y, f"{int(n):,}", ha="right", va="center", color="0.3", fontsize=5.5 if small else 6.5)
        ax.set(xlim=(0, 132), xticks=[0, 50, 100])                   # room for the counts
        ax.set_title(title, fontsize=8, fontweight="bold", pad=3)
        ax.set_xlabel("Failure Ratio [%]", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(True)                # boxed panels
        ax.tick_params(axis="x", labelsize=8.5, length=3)
        ax.tick_params(axis="y", labelsize=6.5 if small else 8.5)
        ax.grid(axis="x")
    return fig, {panel: {lab: [int(r.runs), int(r.failing), round(r.failure_pct, 1)] for lab, r in t.iterrows()}
                 for panel, t in fail.items()}


def batch(one_gpu):
    """Fig. 4: median throughput vs. batch size on one GPU, per GPU type (one_gpu: label -> median and runs)."""
    fig, ax = plt.subplots(figsize=COLUMN, layout="constrained")
    for (lab, s), color, marker in zip(one_gpu.items(), [BLUE, VERMILION, GREEN, ORANGE], "os^D"):
        ax.plot(s.index, s["median"], color=color, marker=marker, ms=4, lw=1.2, label=lab)
    log2_axis(ax, [1, 2, 4, 8, 16, 32, 64, 128])
    ax.set(xlim=(0.8, 160), ylim=(0, 10_500), yticks=range(0, 10_001, 2_500),
           xlabel="Batch size [#]", ylabel="Median throughput [tokens/s]")
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=7)
    return fig, {lab: {"batch_size": s.index.astype(int).tolist(),
                       "median_tps": s["median"].round().astype(int).tolist(), "runs": s["size"].tolist()}
                 for lab, s in one_gpu.items()}


def scaling(by_method):
    """Fig. 5: median throughput vs. number of GPUs, per method (all valid runs, not matched)."""
    fig, ax = plt.subplots(figsize=COLUMN, layout="constrained")
    for (method, s), color, marker in zip(by_method.items(), [BLUE, VERMILION, GREEN], "os^"):
        ax.plot(s.index, s.values, color=color, marker=marker, ms=4, lw=1.2, label=method)
    log2_axis(ax, [1, 2, 4, 8, 16, 32])
    ax.set(xlim=(0.8, 45), ylim=(0, 36_000), yticks=[0, 10_000, 20_000, 30_000],
           xlabel="Number of GPUs [#]", ylabel="Median throughput [tokens/s]")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
    return fig, {method: {"number_gpus": s.index.astype(int).tolist(), "median_tps": s.round().astype(int).tolist()}
                 for method, s in by_method.items()}


def speedups(gains):
    """Fig. 6: median speedup of matched pairs per optimization (gains: label -> (speedup [%], pairs))."""
    rows = sorted(gains.items(), key=lambda kv: -kv[1][0])
    top = rows[0][1][0]
    fig, ax = plt.subplots(figsize=COLUMN, layout="constrained")
    ax.barh(range(len(rows)), [v for _, (v, _) in rows], color=BLUE, height=0.6)
    for y, (_, (v, _)) in enumerate(rows):
        inside = v > 0.3 * top
        ax.text(v - 0.02 * top if inside else v + 0.02 * top, y, f"+{v:.0f}%", va="center",
                ha="right" if inside else "left", color="white" if inside else "black", fontsize=8)
    ax.set(yticks=range(len(rows)), yticklabels=[lab for lab, _ in rows], xlim=(0, 1.05 * top),
           xlabel="Median speedup of matched pairs [%]")
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=4))
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.tick_params(axis="y", length=0)
    return fig, {lab: [round(v, 1), n] for lab, (v, n) in rows}
