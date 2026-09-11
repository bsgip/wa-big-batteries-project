"""Reading and writing the cleaned datasets under data/processed_data/clean/.

The extraction side's equivalent is data_extraction/catalog.py. The split
matters: extracted/ is written once per field and expensive to reproduce,
clean/ is rewritten cheaply whenever anything in data_processing changes.
"""

import pandas as pd

from tools.io import save_df_to_parquet
from tools.paths import clean_data_dir


def clean_path(name: str):
    return clean_data_dir / f"{name}.parquet"


def is_processed(name: str) -> bool:
    return clean_path(name).exists()


def save_clean(name: str, df: pd.DataFrame) -> None:
    save_df_to_parquet(df, clean_path(name))


def load_clean(name: str) -> pd.DataFrame:
    """Read a cleaned dataset. Never processes - if it's missing, say so
    rather than silently producing one, so a stale or absent clean/ file is
    visible instead of quietly recomputed."""
    path = clean_path(name)
    if not path.exists():
        raise FileNotFoundError(f"{path} doesn't exist - run data_processing/main.py to produce it")
    return pd.read_parquet(path)
