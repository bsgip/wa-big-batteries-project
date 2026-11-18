"""Plots of ideal vs actual battery action (analysis/ideal_response.py)."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.ideal_response import ACTED, ACTIONS, AGREE, HELD, OPPOSITE, OUTCOMES, ideal_vs_actual
from tools.paths import local_plots_dir
from tools.plot_style import save_figure

# one colour and marker per rule, the same in every figure
RULE_STYLES = {
    "rule_1": {"label": "Rule 1", "color": "#0072B2", "marker": "o"},
    "rule_1_2": {"label": "Rule 1+2", "color": "#D55E00", "marker": "D"},
}
# agreement in neutral grey; the three kinds of disagreement in colour (not green/red, which mean charge/discharge)
OUTCOME_COLORS = {AGREE: "#4D4D4D", HELD: "#E69F00", ACTED: "#56B4E9", OPPOSITE: "#CC79A7"}
HEATMAP_CMAP = "Blues"
TEXT_COLOR = "0.15"


def plot_compliance_dumbbell(summary: pd.DataFrame, title: str, filename: str):
    """Compliance rate of Rule 1 and Rule 1+2 per battery, one panel per far window.

    Args:
        summary: From ideal_response.compliance_summary.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.

    Returns:
        (fig, axes).
    """
    codes = summary.index.get_level_values("code").unique()
    far_windows = summary.index.get_level_values("far_window").unique()
    fig, axes = plt.subplots(
        1, len(far_windows), figsize=(4.5 * len(far_windows), 0.5 * len(codes) + 1.5), sharex=True, sharey=True
    )

    for ax, far_window in zip(axes, far_windows, strict=True):
        rates = summary.xs(far_window, level="far_window").reindex(codes) * 100
        rows = np.arange(len(codes))
        ax.hlines(rows, rates["rule_1"], rates["rule_1_2"], color="0.7", lw=2, zorder=1)
        for rule, style in RULE_STYLES.items():
            ax.scatter(rates[rule], rows, s=60, color=style["color"], marker=style["marker"],
                       edgecolor="white", linewidth=1, label=style["label"], zorder=2)

        ax.set_title(f"Far window {far_window} ahead", fontsize=10)
        ax.set_xlabel("Trading intervals where ideal = actual (%)")
        ax.grid(axis="x", alpha=0.3)

    axes[0].set_yticks(np.arange(len(codes)), codes)
    axes[0].invert_yaxis()  # first battery at the top

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    save_figure(fig, local_plots_dir / filename)
    return fig, axes


def plot_outcome_bars(shares_by_rule: dict[str, pd.DataFrame], title: str, filename: str):
    """Stacked bar of the four outcomes per battery; one row of panels per rule, one column per far window.

    Args:
        shares_by_rule: {rule label: frame from ideal_response.outcome_shares}, drawn top to bottom.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.

    Returns:
        (fig, axes).
    """
    first = next(iter(shares_by_rule.values()))
    codes = first.index.get_level_values("code").unique()
    far_windows = first.index.get_level_values("far_window").unique()
    nrows, ncols = len(shares_by_rule), len(far_windows)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, (0.4 * len(codes) + 1) * nrows + 1), sharex=True, sharey=True, squeeze=False
    )
    rows = np.arange(len(codes))

    for row_axes, (rule_label, shares) in zip(axes, shares_by_rule.items(), strict=True):
        for ax, far_window in zip(row_axes, far_windows, strict=True):
            pct = shares.xs(far_window, level="far_window").reindex(codes) * 100
            left = np.zeros(len(codes))
            for name in OUTCOMES:
                ax.barh(rows, pct[name], left=left, height=0.7, color=OUTCOME_COLORS[name],
                        edgecolor="white", linewidth=1, label=name)
                left += pct[name].to_numpy()

            ax.set_title(f"{rule_label}, far window {far_window} ahead", fontsize=10)
            ax.set_xlim(0, 100)
            ax.grid(axis="x", alpha=0.3)
            ax.set_axisbelow(True)

    axes[0, 0].set_yticks(rows, codes)
    axes[0, 0].invert_yaxis()  # first battery at the top
    for ax in axes[-1, :]:
        ax.set_xlabel("Share of trading intervals (%)")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(handles), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    save_figure(fig, local_plots_dir / filename)
    return fig, axes


def draw_action_heatmap(ax, counts: pd.DataFrame):
    """Draw one 3x3 ideal vs actual table on ax, coloured by share of its row, labelled % and count."""
    shares = counts.div(counts.sum(axis=1), axis=0)
    image = ax.imshow(shares, cmap=HEATMAP_CMAP, vmin=0, vmax=1)

    for i, ideal in enumerate(ACTIONS):
        for j, actual in enumerate(ACTIONS):
            share = shares.loc[ideal, actual]
            color = "white" if share > 0.6 else TEXT_COLOR
            ax.text(j, i, f"{share:.0%}\n({counts.loc[ideal, actual]:,})", ha="center", va="center",
                    fontsize=8, color=color)

    labels = [action.capitalize() for action in ACTIONS]
    ax.set_xticks(range(len(ACTIONS)), labels)
    ax.set_yticks(range(len(ACTIONS)), labels)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Ideal")
    return image


def plot_ideal_vs_actual(tables: pd.DataFrame, rule: str, far_window: str, title: str, filename: str, ncols: int = 3):
    """Ideal vs actual action as a 3x3 heatmap per battery, for one rule and far window.

    Each row (ideal action) is coloured by its own shares, so batteries that charge and
    discharge in different proportions can be compared.

    Args:
        tables: decision_table of each battery, concatenated with a "code" index level.
        rule: "ideal_rule_1" or "ideal_rule_1_2".
        far_window: Far window label, e.g. "1-4h".
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        ncols: Panels per row.

    Returns:
        (fig, axes).
    """
    codes = tables.index.get_level_values("code").unique()
    nrows = -(-len(codes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 3.4 * nrows), squeeze=False)

    for ax, code in zip(axes.flat, codes, strict=False):
        rows = tables.loc[code]
        rows = rows[(rows["far_window"] == far_window) & ~rows["skipped"]]
        image = draw_action_heatmap(ax, ideal_vs_actual(rows, rule))
        ax.set_title(code, fontsize=10)
    for ax in axes.flat[len(codes):]:
        ax.set_visible(False)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 0.9, 1))
    colorbar_ax = fig.add_axes((0.92, 0.15, 0.015, 0.7))
    fig.colorbar(
        image, cax=colorbar_ax, format=lambda x, _: f"{x:.0%}", label="Share of the ideal action's trading intervals"
    )

    save_figure(fig, local_plots_dir / filename)
    return fig, axes


def plot_compliance_by_time_of_day(by_time: pd.DataFrame, far_window: str, title: str, filename: str, ncols: int = 3):
    """Compliance rate of Rule 1 and Rule 1+2 through the day, one panel per battery, for one far window.

    Args:
        by_time: From ideal_response.compliance_by_time_of_day.
        far_window: Far window label, e.g. "1-4h".
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        ncols: Panels per row.

    Returns:
        (fig, axes).
    """
    codes = by_time.index.get_level_values("code").unique()
    nrows = -(-len(codes) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.2 * nrows), sharex=True, sharey=True, squeeze=False)

    for ax, code in zip(axes.flat, codes, strict=False):
        rates = by_time.loc[(code, far_window)] * 100
        for rule, style in RULE_STYLES.items():
            ax.plot(rates.index, rates[rule], color=style["color"], lw=2, label=style["label"])
        ax.set_title(code, fontsize=10)
        ax.set_xticks(range(0, 25, 6), [f"{hour:02d}:00" for hour in range(0, 25, 6)])  # includes the 18:00 peak
        ax.set_xticks(range(0, 25, 3), minor=True)
        ax.set_xlim(0, 24)
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(codes):]:
        ax.set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Ideal = actual (%)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Trading interval starting")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(handles), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    save_figure(fig, local_plots_dir / filename)
    return fig, axes
