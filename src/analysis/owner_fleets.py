import pandas as pd

from data_processing.soc import add_fleet_soc_columns


# Batteries per owner. Neoen's are registered under "Collie Battery Pty Ltd ATF Collie Battery Trust"
# (data/registration/facilities.csv). ALINTA_WGP_ESR1 is left out, as in the rest of the fleet analysis.
OWNER_BATTERIES = {
    "Neoen": ["COLLIE_ESR1", "COLLIE_BESS2"],
    "Synergy": ["KWINANA_ESR1", "KWINANA_ESR2", "COLLIE_ESR4", "COLLIE_ESR5"],
}


def owner_fleet_soc_pct(
    soc: pd.DataFrame,
    commissioning: dict[str, pd.Timestamp],
    owners: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Fleet SOC per owner, computed the same way as the whole-fleet fleet_soc_pct.

    Args:
        soc: Clean SOC frame (per-battery MWh columns).
        commissioning: {code: commissioning timestamp}, from constants.commissioning_dates on
            the full history, so batteries not reporting yet still count toward capacity.
        owners: {owner: [battery codes]}; defaults to OWNER_BATTERIES.

    Returns:
        One row per interval, one SOC % column per owner.
    """
    owners = OWNER_BATTERIES if owners is None else owners
    out = {}
    for owner, codes in owners.items():
        owner_fleet = add_fleet_soc_columns(soc[codes].copy(), commissioning=commissioning, codes=codes)
        out[owner] = owner_fleet["fleet_soc_pct"]
    return pd.DataFrame(out)
