"""Cleaning for system demand and underlying demand (MW), from the extracted dispatch_condition frame.

process() -> clean/demand.parquet, process_underlying_demand() -> clean/underlying_demand.parquet.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEMAND_COLUMN = "dispatchCondition.demand"
UNDERLYING_DEMAND_COLUMN = "dispatchCondition.underlyingDemand"


def clean_demand(demand: pd.Series) -> pd.Series:
    """Mask exact-zero readings as missing (demand never genuinely hits 0; all zeros are on 2023-12-06)."""
    zeros = demand == 0
    if zeros.any():
        logger.info(f"{demand.name}: masking {zeros.sum()} zero readings as missing")
    return demand.mask(zeros)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Extracted dispatch_condition frame -> one "demand" column (MW), zero readings masked as NaN."""
    return clean_demand(df[DEMAND_COLUMN]).rename("demand").to_frame()


def process_underlying_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Extracted dispatch_condition frame -> one "underlying_demand" column (MW), zero readings masked as NaN."""
    return clean_demand(df[UNDERLYING_DEMAND_COLUMN]).rename("underlying_demand").to_frame()
