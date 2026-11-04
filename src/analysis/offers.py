"""Battery energy offers: the price they discharge at, and whether it fits an opportunity-cost explanation.
Prices are lfa_price (loss-factor adjusted); uses clean/bidstack, without the ±99,999 placeholders.
"""

import pandas as pd

from data_extraction.catalog import load
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START
from tools.io import load_clean


def discharge_offers(codes: list[str]) -> pd.DataFrame:
    """Discharge tranches (positive quantity) of the given batteries, in-service and available.

    Args:
        codes: Battery codes.

    Returns:
        One row per tranche: dispatch_interval, code, quantity (MW) and lfa_price ($/MWh).
    """
    bids = load_clean("bidstack", start_date=None).reset_index()  # every offer, not just the analysis period
    bids = bids.loc[bids["code"].isin(codes), ["dispatch_interval", "code", "quantity", "lfa_price"]]
    return bids.loc[bids["quantity"] > 0]


def weighted_offer_price(offers: pd.DataFrame) -> pd.DataFrame:
    """MW-weighted average discharge offer price of each battery.

    Args:
        offers: Discharge tranches, from discharge_offers.

    Returns:
        One row per interval, one $/MWh column per battery; NaN where it offered no discharge.
    """
    value = offers["quantity"] * offers["lfa_price"]
    by = [offers["dispatch_interval"], offers["code"]]
    weighted = value.groupby(by).sum() / offers["quantity"].groupby(by).sum()
    return weighted.unstack()


def cheapest_offer_price(offers: pd.DataFrame) -> pd.DataFrame:
    """Cheapest discharge offer price of each battery (the price at which it starts to discharge).

    Args:
        offers: Discharge tranches, from discharge_offers.

    Returns:
        One row per interval, one $/MWh column per battery; NaN where it offered no discharge.
    """
    return offers.groupby(["dispatch_interval", "code"])["lfa_price"].min().unstack()


def esroi_opportunity_benchmark() -> pd.Series:
    """Highest forecast price for the rest of the ESROI, for each pre-dispatch run made during it.

    Returns:
        $/MWh, indexed by the run's primary dispatch interval. Runs in the last ESROI
        half-hour have no value.
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
    """Cheapest discharge offer minus the opportunity benchmark, per battery and ESROI interval.

    Each 5-min interval uses the run made at the start of its half-hour. Above 0 means the
    battery only discharges above the best price it could expect later in the ESROI.

    Args:
        cheapest: Cheapest offer price per battery, from cheapest_offer_price.
        benchmark: From esroi_opportunity_benchmark.

    Returns:
        One row per ESROI interval with a benchmark, one $/MWh column per battery.
    """
    run = cheapest.index.floor("30min")
    return cheapest.sub(benchmark.reindex(run).to_numpy(), axis=0).dropna(how="all")
