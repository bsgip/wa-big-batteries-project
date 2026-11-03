"""Who sets the market price?

Each dispatch interval, the dispatch solution lists, per market service, the variables whose price
set that service's price - its "contributions". Most are offer tranches (<code>_<service>_Tranche_<n>,
e.g. KWINANA_ESR1_energy_Tranche_01 or KWINANA_ESR1_contingencyLower_Tranche_02 - energy and FCESS
are co-optimised, so FCESS offers can help set the energy price); the rest are deficit or network
constraint variables (e.g. contRaise_deficit). A price often has several contributions.

A price setter here is a facility (any of its tranches), or a non-offer variable, counted once per
interval however many of its tranches contribute.
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
    """One row per contribution to `service`'s price: dispatch_interval, facility and offer_service
    (the service the tranche is offered in) - both NaN for variables that aren't offer tranches."""
    df = load("price_setting", columns=["market_service", "contributing_variable"])
    df = df.loc[df["market_service"] == service].dropna(subset=["contributing_variable"])

    parts = df["contributing_variable"].str.extract(rf"^(.+?)_({'|'.join(SERVICES)})_Tranche_\d+$")
    return pd.DataFrame({"facility": parts[0], "offer_service": parts[1]}).reset_index()


def monthly_price_setter_shares(codes: list[str], service: str = "energy") -> pd.DataFrame:
    """Each month, the share (%) of all of `service`'s price setters that each battery accounts for.

    Per interval, each price setter counts once; the denominator is the number of price setters
    summed over the month's intervals, so a month adds up to 100%.

    codes: battery codes, one column each; every other facility goes into OTHER_FACILITIES, and
        variables that aren't offer tranches into NON_OFFER
    service: market service, e.g. "energy", "contingencyRaise"

    Returns one row per month, columns: codes, then OTHER_FACILITIES and NON_OFFER.
    """
    c = price_setting_contributions(service)
    setter = c["facility"].where(c["facility"].isin(codes), OTHER_FACILITIES).where(c["facility"].notna(), NON_OFFER)

    setters = c[["dispatch_interval"]].assign(setter=setter).drop_duplicates()  # each setter once per interval
    counts = setters.groupby([pd.Grouper(key="dispatch_interval", freq="MS"), "setter"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=[*codes, OTHER_FACILITIES, NON_OFFER], fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0) * 100


def monthly_battery_energy_offer_types(codes: list[str]) -> pd.DataFrame:
    """Each month, across the battery fleet: when a battery sets the energy price, which of its
    offers do - its energy offer only, energy and FCESS offers, or FCESS offers only.

    Counted per battery per interval (a battery that sets the price counts once in that interval);
    the denominator is all such battery-intervals in the month, so a month adds up to 100%.

    codes: the batteries in the fleet

    Returns one row per month, columns ENERGY_ONLY, ENERGY_AND_FCESS, FCESS_ONLY.
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
