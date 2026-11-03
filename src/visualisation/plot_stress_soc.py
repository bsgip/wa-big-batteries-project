import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tools.paths import local_plots_dir
from tools.plot_style import save_figure


def plot_soc_moments_stress_vs_other(soc_pct, stress_days, moments, title, filename,
                                     stress_color="tab:red", whis=(10, 90)):
    """One panel per definition: fleet SOC at a few moments of the day - a box for all other
    days at each moment, and each stress day as a dot on top of it.

    soc_pct: fleet SOC (%), 5-min
    stress_days: {definition label: DatetimeIndex of stress days (midnight)} - days outside soc_pct are ignored
    moments: {label: "HH:MM"}, e.g. {"Midday\n12:00": "12:00", ...}, drawn left to right
    whis: whisker percentiles for the other-days boxes (default 10th-90th, as in period_box_plot)

    Returns (fig, axes, summary) - summary has, per definition and moment, the median SOC
    on stress days and on other days.
    """
    at = {label: soc_pct.at_time(t).set_axis(soc_pct.at_time(t).index.normalize()) for label, t in moments.items()}

    labels = list(stress_days)
    fig, axes = plt.subplots(1, len(labels), figsize=(5.5 * len(labels), 5), sharey=True, squeeze=False)
    axes = axes[0]

    rows = []
    for ax, label in zip(axes, labels, strict=True):
        for x, (moment, values) in enumerate(at.items()):
            is_stress = values.index.isin(stress_days[label])
            other, stress = values[~is_stress].dropna(), values[is_stress].dropna()

            ax.boxplot(
                [other.to_numpy()], positions=[x], widths=0.5, whis=whis, manage_ticks=False,
                patch_artist=True,
                boxprops={"facecolor": "0.8", "alpha": 0.8},
                medianprops={"color": "black", "lw": 2},
                whiskerprops={"color": "0.4", "lw": 1},
                capprops={"color": "0.4", "lw": 1},
                flierprops={"marker": "o", "markersize": 2.5, "markerfacecolor": "0.5",
                            "markeredgecolor": "none", "alpha": 0.6},
            )
            # stress days spread a little sideways so dots on the same value don't hide each other
            offsets = [((i % 5) - 2) * 0.05 for i in range(len(stress))]
            ax.scatter([x + o for o in offsets], stress.to_numpy(), s=30, color=stress_color,
                       edgecolor="white", lw=0.6, zorder=3)
            rows.append({"definition": label, "moment": moment.replace("\n", " "),
                         "stress_days": len(stress), "stress_median": stress.median(), "other_median": other.median()})

        n = int(at[next(iter(at))].index.isin(stress_days[label]).sum())
        ax.set_title(f"{label} ({n} stress days)", fontsize=10)
        ax.set_xticks(range(len(moments)), labels=list(moments))
        ax.set_xlim(-0.6, len(moments) - 0.4)
        ax.grid(axis="y", alpha=0.3)

    axes[0].set_ylabel("Fleet SOC (%)")
    axes[0].set_ylim(0, 100)

    lo, hi = whis
    handles = [
        Line2D([], [], ls="none", marker="o", markersize=6, color=stress_color, label="Stress day"),
        Patch(facecolor="0.8", edgecolor="black", label="Other days: p25–p75"),
        Line2D([], [], color="black", lw=2, label="Other days: median"),
        Line2D([], [], color="0.4", lw=1, label=f"Other days: p{lo}–p{hi} (whiskers)"),
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.95), ncol=len(handles),
               fontsize=9, frameon=False)
    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.9))

    save_figure(fig, local_plots_dir / filename)

    return fig, axes, pd.DataFrame(rows).set_index(["definition", "moment"])
