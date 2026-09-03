import math

import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, battery_codes
from tools.paths import repo_plots_dir
from tools.plot_style import save_figure


def _grid_shape(n: int) -> tuple[int, int]:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    return rows, cols


def _time_str_to_hour(t: str) -> float:
    hour, minute = t.split(":")
    return int(hour) + int(minute) / 60


# x-axis here is hour-of-day (float 0-24), not a real timestamp like in
# plot_soc.py's per-day plots, so the Peak ESROI window is expressed the
# same way rather than as a pd.Timestamp.
_PEAK_START_HOUR = _time_str_to_hour(PEAK_ESROI_START)
_PEAK_END_HOUR = _time_str_to_hour(PEAK_ESROI_END)


def _lookback_slice(df: pd.DataFrame, lookback_days: int | None) -> pd.DataFrame:
    if lookback_days is None:
        return df
    cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
    return df[df.index > cutoff]


def plot_soc_timeofday_fan_grid(
    df: pd.DataFrame,
    column_suffix: str,
    ylabel: str,
    title: str,
    filename: str,
    lookback_days: int | None = 365,
):
    """Grid of one time-of-day fan chart per battery: median line with
    25-75% and 5-95% shaded bands, built from the `<code><column_suffix>`
    columns of `df`, binned to 5-min-of-day resolution across the last
    `lookback_days` of data (None uses the whole df)."""
    df = _lookback_slice(df, lookback_days)
    hour = df.index.hour + df.index.minute / 60

    quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows, cols = _grid_shape(len(battery_codes))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, code in enumerate(battery_codes):
        ax = axes[i // cols][i % cols]
        ax.axvspan(
            _PEAK_START_HOUR,
            _PEAK_END_HOUR,
            color="gold",
            alpha=0.2,
            zorder=0,
            label=f"Peak ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        )

        col = f"{code}{column_suffix}"
        if col in df:
            long_df = pd.DataFrame({"hour": hour, "value": df[col].to_numpy()}).dropna()
            if not long_df.empty:
                q = long_df.groupby("hour")["value"].quantile(quantiles).unstack()
                ax.fill_between(q.index, q[0.05], q[0.95], color="tab:blue", alpha=0.15, label="5th-95th percentile")
                ax.fill_between(q.index, q[0.25], q[0.75], color="tab:blue", alpha=0.35, label="25th-75th percentile")
                ax.plot(q.index, q[0.5], color="tab:blue", linewidth=1.5, label="median")

        ax.set_title(code)
        ax.set_xlabel("time of day (hour)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 4))

    for i in range(len(battery_codes), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.975), fontsize="small")

    fig.suptitle(title, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save_figure(fig, repo_plots_dir / filename)


def plot_soc_timeofday_box_grid(
    df: pd.DataFrame,
    column_suffix: str,
    ylabel: str,
    title: str,
    filename: str,
    lookback_days: int | None = 365,
):
    """Grid of one time-of-day box plot per battery, one box per hour,
    built from the `<code><column_suffix>` columns of `df`, over the last
    `lookback_days` of data (None uses the whole df)."""
    df = _lookback_slice(df, lookback_days)
    hour_bin = df.index.hour

    rows, cols = _grid_shape(len(battery_codes))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for i, code in enumerate(battery_codes):
        ax = axes[i // cols][i % cols]
        ax.axvspan(
            _PEAK_START_HOUR,
            _PEAK_END_HOUR,
            color="gold",
            alpha=0.2,
            zorder=0,
            label=f"Peak ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        )

        col = f"{code}{column_suffix}"
        if col in df:
            long_df = pd.DataFrame({"hour": hour_bin, "value": df[col].to_numpy()}).dropna()
            if not long_df.empty:
                groups = [long_df.loc[long_df["hour"] == h, "value"] for h in range(24)]
                ax.boxplot(
                    groups,
                    positions=range(24),
                    widths=0.6,
                    showfliers=False,
                    patch_artist=True,
                    boxprops={"facecolor": "tab:blue", "alpha": 0.5},
                    medianprops={"color": "black"},
                )

        ax.set_title(code)
        ax.set_xlabel("time of day (hour)")
        ax.set_ylabel(ylabel)
        ax.set_xlim(-1, 24)
        ax.set_xticks(range(0, 24, 4))

    for i in range(len(battery_codes), rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 0.975), fontsize="small")

    fig.suptitle(title, y=1.005)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    save_figure(fig, repo_plots_dir / filename)
