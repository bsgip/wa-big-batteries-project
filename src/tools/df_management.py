from pathlib import Path

import pandas as pd

from tools.constants import battery_capacity_MWh


CHARGE_LEVEL_SENTINEL = 999


def save_df_to_csv(df: pd.DataFrame, filename: Path):
    df.to_csv(filename)
    print(f"csv successfully saved to {filename}")


def clean_charge_level_df(df: pd.DataFrame) -> pd.DataFrame:
    """The chargeLevel SCADA tag reports a fixed 999 (MWh) sentinel when
    telemetry is missing/invalid (e.g. KWINANA_ESR2 sits at exactly 999 for
    long stretches, well above its 900 MWh capacity). Treat it as missing
    data rather than a real reading."""
    return df.replace(CHARGE_LEVEL_SENTINEL, float("nan"))


def add_soc_pct_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add a <code>_soc_pct column for each battery, computed from its
    charge_level column (MWh) and rated capacity in battery_capacity_MWh."""
    for code, capacity_mwh in battery_capacity_MWh.items():
        df[f"{code}_soc_pct"] = df[code] / capacity_mwh * 100

    return df
