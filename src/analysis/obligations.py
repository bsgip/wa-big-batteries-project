"""Do batteries offer their capacity obligation (RCOQ)?

A battery meets its obligation in a dispatch interval by offering (bidding)
at least its RCOQ as injection. Offered injection is the sum of its positive
tranche quantities: negative tranches are offers to charge, and adding them
in would let a charging offer cancel out discharge offered in the same interval.
"""

import pandas as pd

from data_extraction.catalog import load
from tools.constants import battery_codes

# float noise: 199.98 MW offered against a 200 MW RCOQ is not a shortfall worth counting
SHORTFALL_TOLERANCE_MW = 0.01


def offered_injection_mw(codes: list[str] | None = None) -> pd.DataFrame:
    """Injection (MW) each battery offered per dispatch interval: the sum of its positive tranches,
    IN-SERVICE and AVAILABLE alike.

    NaN where a battery has no offer at all that interval - a gap in the offer data,
    not an offer of 0.
    codes: batteries to keep (default: battery_codes)
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
    """True where a battery has an obligation (RCOQ > 0), made an offer, and offered less than its RCOQ.

    offered: from offered_injection_mw
    rcoq: load_clean("rcoq"), same columns and interval index
    """
    offered = offered.reindex(index=rcoq.index, columns=rcoq.columns)
    return (rcoq > 0) & offered.notna() & (offered < rcoq - tolerance)
