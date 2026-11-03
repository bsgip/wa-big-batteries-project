import matplotlib.pyplot as plt

from tools.paths import local_plots_dir
from tools.plot_style import DISCHARGE_COLOR, save_figure

# one colour per part of the day, in DAY_WINDOWS order; the evening (ESROI) slice in the discharge red
WINDOW_COLORS = ["C0", "C1", "C2", DISCHARGE_COLOR]


def plot_cycles_by_window(cycles, owners, title, filename):
    """One stacked bar per battery: average equivalent full cycles per day, split by part of the day.

    cycles: from energy.discharge_cycles_by_window - one row per battery, one column per window
    owners: {code: owner name}, written under each battery's name; batteries are drawn in the
        order of this mapping

    Returns (fig, ax).
    """
    codes = [code for code in owners if code in cycles.index]
    cycles = cycles.loc[codes]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bottom = [0.0] * len(codes)
    for color, window in zip(WINDOW_COLORS, cycles.columns, strict=True):
        ax.bar(range(len(codes)), cycles[window], bottom=bottom, width=0.6, color=color, alpha=0.85,
               edgecolor="white", lw=1.5, label=window)
        bottom = [b + v for b, v in zip(bottom, cycles[window], strict=True)]

    # total cycles per day on top of each bar
    for x, total in enumerate(bottom):
        ax.text(x, total + 0.02, f"{total:.2f}", ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(range(len(codes)), labels=[f"{code}\n({owners[code]})" for code in codes], fontsize=9)
    ax.set_ylim(0, max(bottom) * 1.15)
    ax.set_ylabel("Equivalent full cycles per day\n(average discharge / capacity)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(cycles.columns), fontsize=9, frameon=False)
    ax.set_title(title, pad=28)  # room for the legend row

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)

    return fig, ax
