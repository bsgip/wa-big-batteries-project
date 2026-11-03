"""Battery energy offers: what price they discharge at, and whether that price can be explained as an
opportunity cost under the Offer Construction Guidelines.

Offers come from the bidstack: per battery and dispatch interval, price-quantity tranches, positive
quantity = discharge (injection), negative = charge (withdrawal). Prices used are lfa_price: the
submitted price adjusted by the battery's loss factor and clipped to the market price limits of the
time - the price the market compares with the clearing price. Uses clean/bidstack, which drops the
±99,999 placeholder prices (see data_processing/bidstack.py).
"""

import pandas as pd

from data_extraction.catalog import load
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START
from tools.io import load_clean


def discharge_offers(codes: list[str]) -> pd.DataFrame:
    """Discharge tranches (positive quantity) of the given batteries: dispatch_interval, code,
    quantity (MW) and lfa_price ($/MWh), IN-SERVICE and AVAILABLE alike."""
    bids = load_clean("bidstack", start_date=None).reset_index()  # every offer, not just the analysis period
    bids = bids.loc[bids["code"].isin(codes), ["dispatch_interval", "code", "quantity", "lfa_price"]]
    return bids.loc[bids["quantity"] > 0]


def weighted_offer_price(offers: pd.DataFrame) -> pd.DataFrame:
    """Per interval, each battery's MW-weighted average discharge offer price ($/MWh): the average
    price of its offered MW. One column per battery; NaN where it offered no discharge."""
    value = offers["quantity"] * offers["lfa_price"]
    by = [offers["dispatch_interval"], offers["code"]]
    weighted = value.groupby(by).sum() / offers["quantity"].groupby(by).sum()
    return weighted.unstack()


def cheapest_offer_price(offers: pd.DataFrame) -> pd.DataFrame:
    """Per interval, each battery's cheapest discharge offer price ($/MWh) - the price at which it
    starts to discharge. One column per battery; NaN where it offered no discharge."""
    return offers.groupby(["dispatch_interval", "code"])["lfa_price"].min().unstack()


def esroi_opportunity_benchmark() -> pd.Series:
    """Per pre-dispatch run made during the ESROI: the highest forecast price over the rest of that
    day's ESROI - the half-hours after the one the run is made in, up to the last that starts
    before the ESROI ends.

    This is a battery's best alternative for discharging later in the ESROI, as it could be seen at
    the time (the guideline's opportunity cost, from AEMO's forecast). Runs in the last half-hour
    have no later ESROI half-hour, so no benchmark.

    Returns a Series indexed by the run's primary dispatch interval (half-hour start).
    """
    forecast = load("predispatch_price")
    start, end = pd.Timedelta(f"{PEAK_ESROI_START}:00"), pd.Timedelta(f"{PEAK_ESROI_END}:00")
    time_of_day = forecast["dispatch_interval"] - forecast["dispatch_interval"].dt.normalize()
    run_time_of_day = forecast["primary_dispatch_interval"] - forecast["primary_dispatch_interval"].dt.normalize()

    later_in_esroi = (
        (run_time_of_day >= start)  # runs made during the ESROI
        & (forecast["dispatch_interval"] > forecast["primary_dispatch_interval"])
        & (forecast["dispatch_interval"].dt.normalize() == forecast["primary_dispatch_interval"].dt.normalize())
        & (time_of_day >= start)
        & (time_of_day < end)  # half-hours starting inside the ESROI
    )
    return forecast.loc[later_in_esroi].groupby("primary_dispatch_interval")["energy_price"].max().rename("benchmark")


def esroi_offer_gap(cheapest: pd.DataFrame, benchmark: pd.Series) -> pd.DataFrame:
    """Per battery, per ESROI interval with a benchmark: its cheapest discharge offer minus the
    highest price forecast for the rest of the ESROI ($/MWh). Above 0: the battery only starts to
    discharge at a price above the best it could expect later - hard to explain as opportunity cost.

    Each 5-min interval uses the latest run: the one made at the start of its half-hour.

    cheapest: from cheapest_offer_price
    benchmark: from esroi_opportunity_benchmark

    Returns one row per interval (ESROI, with a benchmark), one column per battery.
    """
    run = cheapest.index.floor("30min")
    return cheapest.sub(benchmark.reindex(run).to_numpy(), axis=0).dropna(how="all")
