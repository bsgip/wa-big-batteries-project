"""Bidstack cleaning: extracted/bidstack.parquet -> clean/bidstack.parquet.
Drops the ±99,999 $/MWh placeholder tranches (not real offers; only in Aug-Nov 2024).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

PLACEHOLDER_PRICE = 99_999.0


def drop_placeholder_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Drop tranches whose submitted price is ±PLACEHOLDER_PRICE."""
    placeholder = df["submitted_price"].abs() >= PLACEHOLDER_PRICE
    if placeholder.any():
        dropped = df.loc[placeholder]
        logger.info(
            f"bidstack: dropping {placeholder.sum()} tranches priced ±{PLACEHOLDER_PRICE:,.0f} - "
            f"{sorted(dropped['code'].unique())}, "
            f"{dropped['dispatch_interval'].min()} -> {dropped['dispatch_interval'].max()}"
        )
    return df.loc[~placeholder]


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the extracted bid stack.

    Args:
        df: Extracted bidstack, one row per (dispatch_interval, code, tranche).

    Returns:
        The bid stack without placeholder-price tranches, indexed by dispatch_interval.
    """
    return drop_placeholder_prices(df).set_index("dispatch_interval").sort_index()
