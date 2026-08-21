from operator import index
import pandas as pd
from tools.paths import processed_data_dir

filename = processed_data_dir / "charge_level_data.csv"

df = pd.read_csv(filename, parse_dates=["dispatch_interval"], index_col="dispatch_interval")

alinta = df.loc["2024-12-11", "ALINTA_WGP_ESR1"]
print(alinta)
