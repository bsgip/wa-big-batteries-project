from matplotlib.pylab import percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

from tools.paths import extracted_data_dir, clean_data_dir, local_data_dir, local_plots_dir
from tools.constants import (
    battery_capacity_MW,
    battery_capacity_MWh,
    battery_codes,
    codes_in,
    commissioned_capacity,
    commissioning_dates,
)
from tools.plot_style import UNIT_COLORS, save_figure
from visualisation.plot_distributions import plot_distribution_grid, _plot_one_distribution
from visualisation.plot_fleet_capacity import plot_fleet_capacity_evolution


pd.set_option("display.max_columns", None)

soc_ext_path = extracted_data_dir / "soc.parquet"
soc_clean_path = clean_data_dir / "soc.parquet"
power_clean_path = clean_data_dir / "power.parquet"

soc_ext = pd.read_parquet(soc_ext_path)
soc_clean = pd.read_parquet(soc_clean_path)
power_clean = pd.read_parquet(power_clean_path)
print(power_clean.head())




fleet = soc_clean["fleet_soc_pct"]
pct_cols = [col for col in soc_clean.columns if "soc_pct" in col.lower()]


# print(fleet_net_energy.head())





df = pd.DataFrame(fleet.at_time("17:30"))

# Add list of potential predicotrs/variables
df["soc_yesterday"] = df["fleet_soc_pct"].shift(1)
df["soc_last_week"] = df["fleet_soc_pct"].shift(7)
df["soc_30d_median"] = df["fleet_soc_pct"].rolling("30D", closed="left", min_periods=28).median()
df["soc_7d_median"] = df["fleet_soc_pct"].rolling("7D", closed="left", min_periods=7).median()


# print(df.head(15))
# print(df.iloc[120:135])
