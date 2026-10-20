"""RCOQ cleaning: extracted/rcoq.parquet (every facility) -> clean/rcoq.parquet (fleet batteries only)."""

import pandas as pd

from tools.constants import battery_codes, codes_in


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Extracted RCOQ (one MW column per facility) -> only the fleet battery columns."""
    return df[codes_in(df, battery_codes)]
