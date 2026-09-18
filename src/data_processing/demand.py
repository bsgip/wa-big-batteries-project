"""Cleaning for system demand (the extracted dispatchCondition.demand field,
in MW).

No `process()` here and no clean/demand.parquet: demand needs one cheap
masking rule rather than a full cleaning pass, so callers apply it at read
time (see exploration/delta_peak_trough.py and
exploration/variables_correlations.py) and data_processing/main.py leaves
demand alone.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEMAND_COLUMN = "dispatchCondition.demand"


def clean_demand(demand: pd.Series) -> pd.Series:
    """Mask exact-zero demand readings as missing. System demand never
    genuinely reaches 0 MW - the next lowest reading in the whole corpus is
    above 500 MW - and all 9 zeros sit on a single day (2023-12-06). Left
    in, they make that day's trough 0 and hand it the largest peak-trough
    delta in the entire dataset, ahead of every real summer peak day."""
    zeros = demand == 0
    if zeros.any():
        logger.info(f"demand: masking {zeros.sum()} zero readings as missing")
    return demand.mask(zeros)
