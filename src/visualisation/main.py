import pandas as pd
from tools.df_management import add_soc_pct_columns, clean_charge_level_df, mask_sustained_zero_runs, derive_capacity_from_observed_max
from visualisation.plot_time_of_day import plot_soc_timeofday_fan_grid, plot_soc_timeofday_box_grid
from tools.paths import local_processed_data_dir

raw = pd.read_parquet(local_processed_data_dir / "soc.parquet")
cleaned = clean_charge_level_df(raw)
masked = mask_sustained_zero_runs(cleaned)
capacity = derive_capacity_from_observed_max(masked)
df = add_soc_pct_columns(masked, capacity)

# print(df.head())

plot_soc_timeofday_box_grid(
    df, '_soc_pct', 'SOC (%)',
    'Battery SOC by time of day — full history',
    'soc_timeofday_box_grid.png',
)