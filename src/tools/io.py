"""Save data files, and read/write the cleaned datasets in clean/. Extracted ones: data_extraction.catalog.load."""

from pathlib import Path

import pandas as pd

from tools.paths import clean_data_dir
from tools.constants import START_DATE


def save_df_to_csv(df: pd.DataFrame, filename: Path):
    """Save df to filename as CSV, creating the folder if needed."""
    filename.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(filename)
    print(f"csv successfully saved to {filename}")


def save_df_to_parquet(df: pd.DataFrame, filename: Path):
    """Save df to filename as parquet, creating the folder if needed."""
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
    """Read a cleaned dataset. Never builds it.

    Args:
        name: Dataset name, e.g. "soc".
        start_date: First date to keep; None for the whole timeline.

    Returns:
        The dataset from start_date on.

    Raises:
        FileNotFoundError: If clean/<name>.parquet doesn't exist.
    """
    path = clean_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} doesn't exist - run data_processing/main.py to produce it")
    df = pd.read_parquet(path).sort_index()
    
    if start_date is None:
        return df
    
    return df.loc[start_date:]


