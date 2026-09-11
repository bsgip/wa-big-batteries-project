import math

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import logging

from tools.paths import local_data_dir, local_processed_data_dir, local_plots_dir
from tools.constants import battery_codes


logger = logging.getLogger(__name__)


def mask_sustained_zero_runs(df: pd.DataFrame, min_run_minutes: int = 60) -> pd.DataFrame:
    df = df.copy()
    interval_minutes = df.index.to_series().diff().median().total_seconds() / 60
    min_run_length = max(1, round(min_run_minutes / interval_minutes))

    for code in battery_codes:
        if code not in df:
            continue
        is_zero = df[code] == 0
        run_id = (is_zero != is_zero.shift()).cumsum()
        run_length = is_zero.groupby(run_id).transform("size")
        sustained_zero = is_zero & (run_length >= min_run_length)

        n_masked = sustained_zero.sum()
        if n_masked:
            logger.info(f"{code}: masking {n_masked} sustained-zero readings as missing")
        df.loc[sustained_zero, code] = float("nan")

    return df


def plot_sustained_zero_timeline(raw_df: pd.DataFrame, masked_df: pd.DataFrame):
    """Grid of one time series per battery: the raw SOC (MWh) trace, with
    the points that mask_sustained_zero_runs flagged (raw==0 but now NaN
    in masked_df) highlighted in red - lets you eyeball where the flagged
    artifact periods actually fall in the timeline."""
    cols = math.ceil(math.sqrt(len(battery_codes)))
    rows = math.ceil(len(battery_codes) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 3 * rows), squeeze=False)

    for i, code in enumerate(battery_codes):
        ax = axes[i // cols][i % cols]

        if code not in raw_df:
            ax.set_visible(False)
            continue

        s = raw_df[code]
        ax.plot(s.index, s, color="tab:blue", linewidth=0.5, label="SOC (MWh)")

        sustained_zero = (s == 0) & masked_df[code].isna()
        ax.scatter(
            s.index[sustained_zero], s[sustained_zero],
            color="red", s=8, zorder=3, label="Consecutive zeros",
        )

        ax.set_title(code)
        ax.set_ylabel("SOC (MWh)")
        ax.legend(fontsize="x-small")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fig.suptitle("Consecutive zeros over time")
    fig.tight_layout()
    filename = local_plots_dir / "consecutive_zeros_timeline.png"
    fig.savefig(filename, dpi=200)
    print(f"plot saved to {filename}")


df = pd.read_parquet(local_processed_data_dir / "soc.parquet")
# print(df.describe())
print((df == 0).sum())

masked = mask_sustained_zero_runs(df, 60)

print((masked == 0).sum())

plot_sustained_zero_timeline(df, masked)