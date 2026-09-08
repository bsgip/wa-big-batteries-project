import pandas as pd
from pathlib import Path

pd.set_option("display.max_columns", None)

data_path = Path.cwd() / "data"
filename = data_path / "market-advisories.csv"

print(filename)
df = pd.read_csv(filename, parse_dates=True, encoding="cp1252")
lor_mask = (
    (df["Details"].str.contains("LOR")) |
    (df["Details"].str.contains("LRC"))
)
lor = df.loc[lor_mask]

lor_level = 1
,./
lor_filtered = lor.loc[
    ~(lor["Details"].str.contains(f"LOR{lor_level}")) & 
    (lor["Withdrawn"] == "N")
    ]

print(lor_filtered)


