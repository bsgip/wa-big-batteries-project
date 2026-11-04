"""Who sets the market price? Each interval, the dispatch solution lists the variables that set each service's price.
A price setter is a facility (any of its tranches) or a non-offer variable, counted once per interval.
"""

import pandas as pd

from data_extraction.catalog import load

SERVICES = ["energy", "regulationRaise", "regulationLower", "contingencyRaise", "contingencyLower", "rocof"]
OTHER_FACILITIES = "Other facilities"
NON_OFFER = "Deficit / constraint variables"

# which of a battery's offers set the energy price, in an interval where it is a price setter
ENERGY_ONLY = "Energy offer only"
ENERGY_AND_FCESS = "Energy and FCESS offers"
FCESS_ONLY = "FCESS offers only"


def price_setting_contributions(service: str = "energy") -> pd.DataFrame:
    """Every contribution to a market service's price.

    Args:
        service: Market service, e.g. "energy" or "contingencyRaise".

    Returns:
        One row per contribution: dispatch_interval, facility and offer_service (the service
        the tranche is offered in); both NaN for variables that aren't offer tranches.
    """
    df = load("price_setting", columns=["market_service", "contributing_variable"])
    df = df.loc[df["market_service"] == service].dropna(subset=["contributing_variable"])

    parts = df["contributing_variable"].str.extract(rf"^(.+?)_({'|'.join(SERVICES)})_Tranche_\d+$")
    return pd.DataFrame({"facility": parts[0], "offer_service": parts[1]}).reset_index()


def monthly_price_setter_shares(codes: list[str], service: str = "energy") -> pd.DataFrame:
    """Each month, the share of a service's price setters that each battery accounts for.

    Each price setter counts once per interval.

    Args:
        codes: Battery codes, one column each.
        service: Market service, e.g. "energy".

    Returns:
        One row per month, % columns for codes, OTHER_FACILITIES and NON_OFFER (rows add up to 100).
    """
    c = price_setting_contributions(service)
    setter = c["facility"].where(c["facility"].isin(codes), OTHER_FACILITIES).where(c["facility"].notna(), NON_OFFER)

    setters = c[["dispatch_interval"]].assign(setter=setter).drop_duplicates()  # each setter once per interval
    counts = setters.groupby([pd.Grouper(key="dispatch_interval", freq="MS"), "setter"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[*codes, OTHER_FACILITIES, NON_OFFER], fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def monthly_battery_energy_offer_types(codes: list[str]) -> pd.DataFrame:
    """Each month, which of a battery's offers set the energy price when it sets it.

    Counted once per battery per interval.

    Args:
        codes: The batteries in the fleet.

    Returns:
        One row per month, % columns ENERGY_ONLY, ENERGY_AND_FCESS and FCESS_ONLY (rows add up to 100).
    """
    c = price_setting_contributions("energy")
    c = c.loc[c["facility"].isin(codes)].assign(energy_offer=lambda d: d["offer_service"] == "energy")

    per_battery = c.groupby(["dispatch_interval", "facility"])["energy_offer"].agg(["all", "any"])
    offer_type = per_battery["all"].map({True: ENERGY_ONLY}).fillna(
        per_battery["any"].map({True: ENERGY_AND_FCESS, False: FCESS_ONLY}))

    counts = (offer_type.rename("offer_type").reset_index()
              .groupby([pd.Grouper(key="dispatch_interval", freq="MS"), "offer_type"]).size().unstack(fill_value=0))
    counts = counts.reindex(columns=[ENERGY_ONLY, ENERGY_AND_FCESS, FCESS_ONLY], fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def hourly_price_setter_shares(groups: pd.Series, start: str, end: str, service: str = "energy") -> pd.DataFrame:
    """By hour of day, the share of a service's price setters in each group.

    Each price setter counts once per interval.

    Args:
        groups: {facility code: group}, e.g. from generation_mix.facility_groups; other
            facilities go into OTHER_FACILITIES.
        start: First date, e.g. "2025-12-01".
        end: End date (exclusive).
        service: Market service, e.g. "energy".

    Returns:
        One row per hour (0-23), one % column per group plus NON_OFFER (rows add up to 100).
    """
    c = price_setting_contributions(service)
    tz = c["dispatch_interval"].dt.tz
    c = c.loc[(c["dispatch_interval"] >= pd.Timestamp(start, tz=tz)) & (c["dispatch_interval"] < pd.Timestamp(end, tz=tz))]

    group = c["facility"].map(groups).fillna(OTHER_FACILITIES).where(c["facility"].notna(), NON_OFFER)
    setters = c[["dispatch_interval", "facility"]].assign(group=group).drop_duplicates()  # each setter once per interval
    counts = setters.groupby([setters["dispatch_interval"].dt.hour.rename("hour"), "group"]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100
