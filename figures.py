"""The paper's figures (Figs. 2-6), one function each: the data reproduce.py computed go in; the figure and its plotted
values come out. reproduce.py lists them in FIGURES under the paper's file names; check.py pins each one's page size,
fonts, matplotlib version and plotted values.

The paper's PDFs were drawn with matplotlib 3.11.0 in Times New Roman (regular and bold, embedded as TrueType). With
both, these functions draw the paper's figures operator for operator. Another build of Times New Roman changes only
the embedded font program: Debian's ttf-mscorefonts-installer (font version 2.82, used by the Docker image) has the
same glyphs and advance widths as the paper's font (version 5.01), so its PDFs render pixel for pixel like the
paper's. matplotlib 3.10 lays out text differently (labels move by up to 1.7 pt), and without Times New Roman
matplotlib would silently draw in another font, so render() refuses to draw without it.
"""
import hashlib
from contextlib import contextmanager

import matplotlib
matplotlib.use("Agg")                         # other backends write narrower PDFs than declared
import matplotlib as mpl
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

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
                               "multiverse), then delete matplotlib's font cache (make clean)")


def render(draw, data, pdf):
    """Draw one figure into pdf; return the file's SHA-256 and the plotted values."""
    require_fonts()
    fig, plotted = draw(data)
    fig.savefig(pdf, metadata={"CreationDate": None})
    plt.close(fig)
    return hashlib.sha256(pdf.read_bytes()).hexdigest(), plotted


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
    return fig, data


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
    type. data: GPU label -> batch sizes, median throughputs [tokens/s] and runs, in Table 1 order."""
    xlim, ylim, ticks = (0.8, 128 * 1.25), (0, 10_000), [2 ** k for k in range(8)]      # batch sizes 1 .. 128
    with styled(COLUMN) as fig_class:
        fig, ax = _line_figure(fig_class)
        _lines(ax, data, "batch_size", {"A100-SXM": BLUE, "A100-PCIe": ORANGE, "L40S": CYAN, "H100-PCIe": MAGENTA},
               {"A100-SXM": "o", "A100-PCIe": "s", "L40S": "D", "H100-PCIe": "^"}, xlim, ylim)
        _line_axes(fig, ax, xlim, ylim, ticks, [f"{t:,}" for t in ticks], [0, 2_500, 5_000, 7_500, 10_000],
                   "Batch Size [#]", ncol=4)
    return fig, data


def scaling(data):
    """Fig. 5 (240.04 x 100.8 pt): median throughput of the valid runs per number of GPUs, one line per fine-tuning
    method (not matched). The dashed line marks the node boundary (8 GPUs per node). data: method label -> GPU
    counts, median throughputs [tokens/s] and runs, in legend order."""
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
    return fig, data


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
    throughput of the valid runs [tokens/s] (None: no valid run), the valid runs and the experiments. A cell with
    experiments but no valid run is blank with a grey outline; a cell without experiments is hatched."""
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
    labels = [["" if np.isnan(x) else compact(x) for x in row] for row in m]
    return fig, {**data, "cell_label": labels}


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
    return fig, rows
