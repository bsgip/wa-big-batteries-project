import logging

import pandas as pd

from tools.df_management import add_soc_pct_columns, clean_charge_level_df
from tools.paths import processed_data_dir
from visualisation.plot_soc import plot_soc_for_stress_events

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    filename = processed_data_dir / "charge_level_data.csv"
    df = pd.read_csv(filename, parse_dates=["dispatch_interval"], index_col="dispatch_interval")
    df = clean_charge_level_df(df)
    df = add_soc_pct_columns(df)

    print(df.tail())
