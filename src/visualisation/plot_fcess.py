import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from tools.paths import local_plots_dir
from tools.plot_style import CHARGE_COLOR, DISCHARGE_COLOR, save_figure

SERVICE_COLORS = {
    "contingencyRaise": "#0072B2",
    "contingencyLower": "#56B4E9",
    "regulationRaise": "#CC79A7",
    "regulationLower": "#E8B4D4",
}
SERVICE_LABELS = {
    "contingencyRaise": "Contingency raise",
    "contingencyLower": "Contingency lower",
    "regulationRaise": "Regulation raise",
    "regulationLower": "Regulation lower",
}


def plot_fcess_and_power_by_window(by_owner, title, filename):
    """One column per owner. Top: average MW enabled for each FCESS service, per window of the day.
    Bottom: average MW discharging (up) and charging (down) in the same windows. Rows share a y-axis.

    by_owner: {owner: (enabled, power)} - enabled: one row per window, one column per service (MW);
        power: one row per window, columns "discharging" and "charging" (MW, both positive)

    Returns (fig, axes).
    """
    fig, axes = plt.subplots(2, len(by_owner), figsize=(6 * len(by_owner), 7.5), sharex=True, sharey="row",
                             gridspec_kw={"height_ratios": [1.4, 1]}, squeeze=False)

    for col, (owner, (enabled, power)) in enumerate(by_owner.items()):
        ax_fcess, ax_power = axes[0, col], axes[1, col]
        x = np.arange(len(enabled.index))
        width = 0.8 / len(enabled.columns)
        for i, service in enumerate(enabled.columns):
            ax_fcess.bar(x + (i - (len(enabled.columns) - 1) / 2) * width, enabled[service], width=width * 0.95,
                         color=SERVICE_COLORS[service])
        ax_fcess.set_title(owner, fontsize=11)

        ax_power.bar(x, power["discharging"], width=0.5, color=DISCHARGE_COLOR, alpha=0.8)
        ax_power.bar(x, -power["charging"], width=0.5, color=CHARGE_COLOR, alpha=0.8)
        ax_power.axhline(0, color="black", lw=0.8)
        ax_power.set_xticks(x, labels=[label.replace(" (", "\n(") for label in enabled.index], fontsize=9)

        for ax in (ax_fcess, ax_power):
            ax.grid(axis="y", alpha=0.3)

    axes[0, 0].set_ylabel("FCESS enabled (MW)\nowner total, average over the window")
    axes[1, 0].set_ylabel("Power (MW), owner total\n+ discharging / − charging")

    handles = [Patch(color=SERVICE_COLORS[s], label=SERVICE_LABELS[s]) for s in SERVICE_COLORS]
    handles += [Patch(color=DISCHARGE_COLOR, alpha=0.8, label="Discharging"),
                Patch(color=CHARGE_COLOR, alpha=0.8, label="Charging")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, fontsize=9, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.91))

    save_figure(fig, local_plots_dir / filename)

    return fig, axes
