"""Saving data files, and reading/writing the cleaned datasets in data/processed_data/clean/.

Extracted datasets are read with data_extraction.catalog.load instead.
"""

from pathlib import Path

import pandas as pd

from tools.paths import clean_data_dir
from tools.constants import START_DATE


def save_df_to_csv(df: pd.DataFrame, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename)
    print(f"csv successfully saved to {filename}")


def save_df_to_parquet(df: pd.DataFrame, filename: Path):
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filename)
    print(f"parquet successfully saved to {filename}")


def clean_path(name: str) -> Path:
    """Path of clean/<name>.parquet."""
    return clean_data_dir / f"{name}.parquet"


def is_processed(name: str) -> bool:
    """Whether clean/<name>.parquet exists."""
    return clean_path(name).exists()


def save_clean(name: str, df: pd.DataFrame) -> None:
    """Write df to clean/<name>.parquet."""
    save_df_to_parquet(df, clean_path(name))


def load_clean(name: str, start_date: str | None = START_DATE) -> pd.DataFrame:
    """Read clean/<name>.parquet; raises if it doesn't exist (never builds it).
    Slices the dataframe starting from a start date default of 2024-12-01"""
    path = clean_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} doesn't exist - run data_processing/main.py to produce it")
    df = pd.read_parquet(path).sort_index()
    
    if start_date is None:
        return df
    
    return df.loc[start_date:]


