"""Bidstack cleaning: extracted/bidstack.parquet -> clean/bidstack.parquet.

Drops tranches with a placeholder price (submitted ±99,999 $/MWh) - not real offers. They're far
outside the market price limits (-1,000 to a cap of 738-1,700 $/MWh over the data), and only appear
in Aug-Nov 2024 (COLLIE_ESR1, KWINANA_ESR1, KWINANA_ESR2).

Indexed by dispatch_interval like the other clean frames (one row per tranche, so repeated).
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
    """Extracted bidstack -> placeholder prices dropped, indexed by dispatch_interval."""
    return drop_placeholder_prices(df).set_index("dispatch_interval").sort_index()
