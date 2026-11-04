"""Do batteries offer their capacity obligation (RCOQ)?
Offered injection = sum of positive tranches, so charge offers can't cancel out discharge offers.
"""

import pandas as pd

from data_extraction.catalog import load
from tools.constants import battery_codes

# float noise: 199.98 MW offered against a 200 MW RCOQ is not a shortfall worth counting
SHORTFALL_TOLERANCE_MW = 0.01


def offered_injection_mw(codes: list[str] | None = None) -> pd.DataFrame:
    """MW each battery offered to inject per dispatch interval (sum of its positive tranches).

    Args:
        codes: Batteries to keep; defaults to battery_codes.

    Returns:
        One row per interval, one MW column per battery; NaN where the battery made no offer.
    """
    codes = battery_codes if codes is None else codes
    bids = load("bidstack", columns=["dispatch_interval", "code", "quantity"])
    bids = bids[bids["code"].isin(codes)]

    has_offer = bids.groupby(["dispatch_interval", "code"]).size().unstack().notna()
    injection = bids[bids["quantity"] > 0]
    offered = injection.pivot_table(index="dispatch_interval", columns="code", values="quantity", aggfunc="sum")

    # an offer with no positive tranche is 0 MW offered; no offer at all stays NaN
    offered = offered.reindex(index=has_offer.index, columns=codes).fillna(0)
    return offered.where(has_offer.reindex(columns=codes, fill_value=False))


def below_rcoq(offered: pd.DataFrame, rcoq: pd.DataFrame, tolerance: float = SHORTFALL_TOLERANCE_MW) -> pd.DataFrame:
    """Flag intervals where a battery offered less than its RCOQ.

    Args:
        offered: Offered injection (MW), from offered_injection_mw.
        rcoq: RCOQ (MW), same index and columns, from load_clean("rcoq").
        tolerance: MW below RCOQ still counted as meeting it.

    Returns:
        Boolean frame: True where RCOQ > 0, an offer exists, and offered < RCOQ - tolerance.
    """
    offered = offered.reindex(index=rcoq.index, columns=rcoq.columns)
    return (rcoq > 0) & offered.notna() & (offered < rcoq - tolerance)
