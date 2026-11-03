import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools.paths import local_plots_dir
from tools.plot_style import save_figure
from tools.seasons import SEASONS, season_of


def plot_two_groups_by_season(values, in_group, group_labels, colors, ylabel, title, filename, whis=(10, 90)):
    """Per season, one box for days in a group and one for the rest, side by side.

    values: daily Series to draw the boxes of
    in_group: daily bool Series, True for days in the group (e.g. high-demand days)
    group_labels: (label for the group, label for the rest)
    colors: (colour for the group, colour for the rest)
    whis: whisker percentiles (default 10th-90th, as in period_box_plot); days beyond are dots

    Returns (fig, ax, summary) - summary has each season's day count and median per group.
    """
    data = pd.concat([values.rename("value"), in_group.rename("in_group")], axis=1).dropna()
    data["in_group"] = data["in_group"].astype(bool)
    data["season"] = season_of(data.index)

    fig, ax = plt.subplots(figsize=(11, 5.5))
    rows = []
    for i, season in enumerate(SEASONS):
        d = data[data["season"] == season]
        for offset, flag, color in [(-0.18, True, colors[0]), (0.18, False, colors[1])]:
            group = d.loc[d["in_group"] == flag, "value"]
            ax.boxplot(
                [group.to_numpy()],
                positions=[i + offset],
                widths=0.3,
                whis=whis,
                manage_ticks=False,
                patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.6},
                medianprops={"color": "black", "lw": 2},
                whiskerprops={"color": "0.4", "lw": 1},
                capprops={"color": "0.4", "lw": 1},
                flierprops={"marker": "o", "markersize": 3, "markerfacecolor": color,
                            "markeredgecolor": "none", "alpha": 0.75},
            )
            rows.append({"season": season, "group": group_labels[0] if flag else group_labels[1],
                         "days": len(group), "median": group.median()})

    ax.set_xticks(range(len(SEASONS)), labels=SEASONS)
    ax.set_xlim(-0.6, len(SEASONS) - 0.4)
    ax.set_ylabel(ylabel)
    ax.grid(axis="y", alpha=0.3)

    lo, hi = whis
    handles = [
        Patch(facecolor=colors[0], alpha=0.6, edgecolor="black", label=group_labels[0]),
        Patch(facecolor=colors[1], alpha=0.6, edgecolor="black", label=group_labels[1]),
        Line2D([], [], color="black", lw=2, label="Median"),
        Line2D([], [], color="0.4", lw=1, label=f"p{lo}–p{hi} (whiskers)"),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles),
              fontsize=9, frameon=False)
    ax.set_title(title, pad=28)  # room for the legend row

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)

    return fig, ax, pd.DataFrame(rows).set_index(["season", "group"])


def plot_panels_by_group(panels, groups, order, title, filename, whis=(10, 90)):
    """One panel per measure, side by side; in each, one box per group of days.

    panels: [(daily Series, y-axis label, colour), ...], drawn left to right
    groups: daily Series of group labels (e.g. a price band), same days as the panels
    order: group labels, left to right - each labelled with its day count
    whis: whisker percentiles (default 10th-90th, as in period_box_plot); days beyond are dots

    Returns (fig, axes, summary) - summary has each group's day count and each measure's median.
    """
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 5), squeeze=False)
    axes = axes[0]

    summary = {}
    for ax, (values, ylabel, color) in zip(axes, panels, strict=True):
        data = pd.concat([values.rename("value"), groups.rename("group")], axis=1).dropna()
        counts = data["group"].value_counts()
        for x, group in enumerate(order):
            ax.boxplot(
                [data.loc[data["group"] == group, "value"].to_numpy()], positions=[x], widths=0.55, whis=whis,
                manage_ticks=False, patch_artist=True,
                boxprops={"facecolor": color, "alpha": 0.5},
                medianprops={"color": "black", "lw": 2},
                whiskerprops={"color": "0.4", "lw": 1},
                capprops={"color": "0.4", "lw": 1},
                flierprops={"marker": "o", "markersize": 3, "markerfacecolor": color, "markeredgecolor": "none",
                            "alpha": 0.6},
            )
        ax.set_xticks(range(len(order)), labels=[f"{g}\n({counts.get(g, 0)} days)" for g in order], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        summary[ylabel.split("\n")[0]] = data.groupby("group", observed=True)["value"].median().reindex(order)
    summary = pd.DataFrame(summary)
    summary.insert(0, "days", groups.value_counts().reindex(order))

    lo, hi = whis
    handles = [Line2D([], [], color="black", lw=2, label="Median"),
               Line2D([], [], color="0.4", lw=1, label=f"p{lo}–p{hi} (whiskers)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles), fontsize=9,
               frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    save_figure(fig, local_plots_dir / filename)

    return fig, axes, summary
