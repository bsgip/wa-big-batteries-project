import matplotlib.pyplot as plt

from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def plot_error_by_lead_time(stats, title, filename, xticks=range(0, 25, 4)):
    """Plot median forecast error by lead time with p25-p75 and p10-p90 bands, one panel per window.

    Args:
        stats: {window label: frame from price_forecast.error_by_lead_time}, drawn left to right.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        xticks: Lead times (hours) to tick.

    Returns:
        (fig, axes).
    """
    fig, axes = plt.subplots(1, len(stats), figsize=(6 * len(stats), 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (label, m) in zip(axes, stats.items(), strict=True):
        ax.fill_between(m.index, m["p10"], m["p90"], alpha=0.18, color="C0", lw=0, label="p10–p90")
        ax.fill_between(m.index, m["p25"], m["p75"], alpha=0.30, color="C0", lw=0, label="p25–p75")
        ax.plot(m.index, m["median"], color="C0", lw=2, marker="o", ms=3, label="Median")
        ax.axhline(0, color="black", ls="--", lw=1, label="No error")

        ax.set_title(label, fontsize=10)
        ax.set_xticks(list(xticks))
        ax.set_xlim(0, m.index.max() + 0.25)  # from 0 so the ticks are evenly spaced
        ax.set_xlabel("Lead time (hours before the half-hour)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Forecast − actual price ($/MWh)")

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles),
               fontsize=9, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    save_figure(fig, local_plots_dir / filename)

    return fig, axes


def plot_windows_by_lead_time(stats, column, colors, ylabel, title, filename, xticks=range(0, 25, 4)):
    """Plot one measure by lead time, one line per window of the day, one panel per period.

    Args:
        stats: {period label: {window label: frame from price_forecast.error_by_lead_time}};
            periods are drawn left to right.
        column: Column of those frames to plot, e.g. "median_abs".
        colors: {window label: colour}.
        ylabel: Y-axis label.
        title: Figure title.
        filename: Path under local_plots_dir to save the figure to.
        xticks: Lead times (hours) to tick.

    Returns:
        (fig, axes).
    """
    fig, axes = plt.subplots(1, len(stats), figsize=(6 * len(stats), 5), sharey=True, squeeze=False)
    axes = axes[0]

    for ax, (period, by_window) in zip(axes, stats.items(), strict=True):
        for window, m in by_window.items():
            ax.plot(m.index, m[column], color=colors[window], lw=2, marker="o", ms=3, label=window)

        ax.set_title(period, fontsize=10)
        ax.set_xticks(list(xticks))
        ax.set_xlim(0, max(m.index.max() for m in by_window.values()) + 0.25)  # from 0: evenly spaced ticks
        ax.set_ylim(bottom=0)
        ax.set_xlabel("Lead time (hours before the half-hour)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel(ylabel)

    handles, _ = axes[0].get_legend_handles_labels()
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=len(handles),
               fontsize=9, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    save_figure(fig, local_plots_dir / filename)

    return fig, axes
