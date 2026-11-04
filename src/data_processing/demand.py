"""System and underlying demand (MW) -> clean/demand.parquet and clean/underlying_demand.parquet."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEMAND_COLUMN = "dispatchCondition.demand"
UNDERLYING_DEMAND_COLUMN = "dispatchCondition.underlyingDemand"


def clean_demand(demand: pd.Series) -> pd.Series:
    """Set exact-zero demand readings to NaN (all of them are on 2023-12-06)."""
    zeros = demand == 0
    if zeros.any():
        logger.info(f"{demand.name}: masking {zeros.sum()} zero readings as missing")
    return demand.mask(zeros)


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Clean system demand.

    Args:
        df: Extracted dispatch_condition frame.

    Returns:
        One "demand" column (MW), zeros set to NaN.
    """
    return clean_demand(df[DEMAND_COLUMN]).rename("demand").to_frame()


def process_underlying_demand(df: pd.DataFrame) -> pd.DataFrame:
    """Clean underlying demand.

    Args:
        df: Extracted dispatch_condition frame.

    Returns:
        One "underlying_demand" column (MW), zeros set to NaN.
    """
    return clean_demand(df[UNDERLYING_DEMAND_COLUMN]).rename("underlying_demand").to_frame()
