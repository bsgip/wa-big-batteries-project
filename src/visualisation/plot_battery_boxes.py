import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def _draw_box(ax, values, x, color, width, whis):
    """One box at x, 10th-90th whiskers by default, values beyond them as dots."""
    ax.boxplot(
        [values.dropna().to_numpy()], positions=[x], widths=width, whis=whis, manage_ticks=False,
        patch_artist=True,
        boxprops={"facecolor": color, "alpha": 0.6},
        medianprops={"color": "black", "lw": 2},
        whiskerprops={"color": "0.4", "lw": 1},
        capprops={"color": "0.4", "lw": 1},
        flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": color, "markeredgecolor": "none",
                    "alpha": 0.4, "rasterized": True},
    )


def _owner_medians(values, owners):
    """Each owner's median over all its batteries' values pooled."""
    return pd.Series({owner: pd.concat([values[c] for c in codes]).median() for owner, codes in owners.items()})


def plot_boxes_by_battery(values, owners, owner_colors, ylabel, title, filename, value_format="${:,.0f}/MWh",
                          whis=(10, 90), ylim=None, zero_line=False):
    """One box per battery, coloured by owner; each owner's median over all its batteries' values in
    the legend.

    values: one column per battery (any frequency) - every non-NaN value goes into that battery's box
    owners: {owner: [battery codes]}, drawn in that order, owner by owner
    owner_colors: {owner: colour}
    value_format: how an owner's median is written in the legend
    whis: whisker percentiles (default 10th-90th, as in period_box_plot); values beyond are dots
    ylim: (bottom, top) of the y-axis - dots outside it are cut off (defaults to fitting the data)
    zero_line: dashed line at 0, e.g. for a gap

    Returns (fig, ax, owner_medians).
    """
    codes = [code for codes in owners.values() for code in codes]
    owner_of = {code: owner for owner, codes in owners.items() for code in codes}

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for x, code in enumerate(codes):
        _draw_box(ax, values[code], x, owner_colors[owner_of[code]], width=0.55, whis=whis)
    if zero_line:
        ax.axhline(0, color="black", ls="--", lw=1, zorder=0)

    ax.set_xticks(range(len(codes)), labels=[f"{code}\n({owner_of[code]})" for code in codes], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    owner_medians = _owner_medians(values, owners)
    lo, hi = whis
    handles = [Patch(facecolor=owner_colors[owner], alpha=0.6, label=f"{owner}: median {value_format.format(m)}")
               for owner, m in owner_medians.items()]
    handles += [Line2D([], [], color="black", lw=2, label="Battery median"),
                Line2D([], [], color="0.4", lw=1, label=f"p{lo}–p{hi} (whiskers)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), fontsize=8.5,
              frameon=False)
    ax.set_title(title, pad=28)  # room for the legend row

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)

    return fig, ax, owner_medians


def plot_window_boxes_by_battery(by_window, window_colors, owners, ylabel, title, filename,
                                 value_format=r"\${:,.0f}/MWh", whis=(10, 90), ylim=None):
    """Side-by-side boxes per battery, one per window of the day (e.g. ESROI, outside it), coloured
    by window; each owner's median per window in the legend.

    by_window: {window label: one column per battery} - drawn left to right within each battery
    window_colors: {window label: colour}
    owners: {owner: [battery codes]}, drawn in that order, owner by owner
    value_format: how an owner's median is written in the legend ($ escaped: two in one label would
        otherwise be read as maths)
    whis: whisker percentiles (default 10th-90th); values beyond are dots
    ylim: (bottom, top) of the y-axis - dots outside it are cut off (defaults to fitting the data)

    Returns (fig, ax, owner_medians) - owner_medians has one row per window, one column per owner.
    """
    codes = [code for codes in owners.values() for code in codes]
    owner_of = {code: owner for owner, codes in owners.items() for code in codes}
    width = 0.8 / len(by_window)
    offsets = [(i - (len(by_window) - 1) / 2) * width for i in range(len(by_window))]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    for x, code in enumerate(codes):
        for offset, (window, values) in zip(offsets, by_window.items(), strict=True):
            _draw_box(ax, values[code], x + offset, window_colors[window], width=width * 0.85, whis=whis)

    ax.set_xticks(range(len(codes)), labels=[f"{code}\n({owner_of[code]})" for code in codes], fontsize=9)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)
    if ylim is not None:
        ax.set_ylim(ylim)

    owner_medians = pd.DataFrame({window: _owner_medians(values, owners) for window, values in by_window.items()}).T
    handles = [
        Patch(facecolor=window_colors[window], alpha=0.6,
              label=f"{window}: median " + ", ".join(f"{owner} {value_format.format(m)}" for owner, m in row.items()))
        for window, row in owner_medians.iterrows()
    ]
    lo, hi = whis
    handles += [Line2D([], [], color="black", lw=2, label="Battery median"),
                Line2D([], [], color="0.4", lw=1, label=f"p{lo}–p{hi} (whiskers)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=8.5, frameon=False)
    ax.set_title(title, pad=42)  # room for two legend rows

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)

    return fig, ax, owner_medians
