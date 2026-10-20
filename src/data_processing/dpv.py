"""Cleaning for distributed (rooftop) PV, from the extracted dispatch_condition frame.

process() -> clean/dpv.parquet.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DPV_COLUMN = "dispatchCondition.dpv"

# Before this, AEMO's DPV has a 6-day stuck value (Sep 2024), impossible night readings and a
# ~2000 MW ceiling (until 2024-11-06). Everything earlier is set to NaN.
DPV_CLEAN_FROM = "2024-11-07"


def process(df: pd.DataFrame) -> pd.DataFrame:
    """Extracted dispatch_condition frame -> one "dpv" column (MW): NaN before DPV_CLEAN_FROM, negatives set to 0."""
    dpv = df[DPV_COLUMN].clip(lower=0)

    before = dpv.index < pd.Timestamp(DPV_CLEAN_FROM, tz=dpv.index.tz)
    logger.info(f"dpv: setting {dpv[before].notna().sum()} readings before {DPV_CLEAN_FROM} to NaN")

    return dpv.mask(before).rename("dpv").to_frame()
