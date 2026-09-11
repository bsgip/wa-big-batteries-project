from pathlib import Path

import pandas as pd


def save_df_to_csv(df: pd.DataFrame, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename)
    print(f"csv successfully saved to {filename}")


def save_df_to_parquet(df: pd.DataFrame, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filename)
    print(f"parquet successfully saved to {filename}")
