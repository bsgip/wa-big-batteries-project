import pandas as pd
from data_processing.soc import (
    add_soc_pct_columns,
    clean_charge_level_sentinel_df,
    mask_sustained_zero_runs,
)
import matplotlib.pyplot as plt
from visualisation.plot_time_of_day import plot_soc_timeofday_fan_grid, plot_soc_timeofday_box_grid
from visualisation.plot_distributions import _plot_one_distribution
from tools.paths import extracted_data_dir
from tools.constants import battery_capacity_MWh


def get_soc_df() -> pd.DataFrame:
    raw = pd.read_parquet(extracted_data_dir / "soc.parquet")
    # cleaned = clean_charge_level_df(raw)
    masked = mask_sustained_zero_runs(raw)
    df = add_soc_pct_columns(masked, battery_capacity_MWh)
    return df

df = get_soc_df()
kwin = df["KWINANA_ESR2"]
s = kwin.loc[(kwin > 900) & (kwin != 999)]
fig, ax = plt.subplots(figsize=(10,6))
ax.scatter(s.index, s)
plt.show()

def get_power_df() -> pd.DataFrame():
    pass


def plot_kwinana_distribution():
    df = get_soc_df()
    fig, ax = plt.subplots(figsize=(10,6))
    _plot_one_distribution(ax, df["KWINANA_ESR2"], False)
    ax.set_xlabel("SOC (MWh)")
    ax.set_ylabel("density")
    fig.suptitle("SOC Distribution")
    fig.tight_layout()
    plt.show()