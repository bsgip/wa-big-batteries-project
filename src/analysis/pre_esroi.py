"""Does the fleet start the ESROI low because of high prices beforehand - by discharging early, or
by charging less? Daily measures of the hours before the ESROI, grouped by how high prices got.
"""

from collections.abc import Mapping

import pandas as pd

from analysis.energy import fleet_net_energy_pct

# highest price before the ESROI ($/MWh): band edges, each band includes its lower edge
PRICE_BAND_EDGES = [150, 300]
PRICE_BAND_LABELS = ["Under $150", "$150–300", "Over $300"]


def daily_max_price(price: pd.Series, start: str, end: str) -> pd.Series:
    """Highest 5-min price between start and end, each day ($/MWh)."""
    return price.between_time(start, end, inclusive="left").resample("D").max()


def price_band(price: pd.Series, edges: list[float] = PRICE_BAND_EDGES,
               labels: list[str] = PRICE_BAND_LABELS) -> pd.Series:
    """Each day's band, from a daily price (e.g. its highest or average price before the ESROI).

    edges: inner band edges ($/MWh), each band includes its lower edge
    labels: one more label than edges, lowest band first
    """
    return pd.cut(price, [-float("inf"), *edges, float("inf")], labels=labels, right=False)


def fleet_charged_discharged_pct(power: pd.DataFrame, commissioning: Mapping[str, pd.Timestamp],
                                 start: str, end: str) -> pd.DataFrame:
    """Daily energy the fleet charged and discharged between start and end, each as % of
    commissioned fleet capacity (MWh) - not MWh, so the fleet growing doesn't read as more of either.
    Both positive; charging and discharging are counted separately, not netted.

    commissioning: from constants.commissioning_dates(...)
    """
    discharged = fleet_net_energy_pct(power.clip(lower=0), commissioning, start, end)
    charged = fleet_net_energy_pct(-power.clip(upper=0), commissioning, start, end)
    return pd.DataFrame({"charged": charged, "discharged": discharged})
