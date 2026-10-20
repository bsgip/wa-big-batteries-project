"""Reserve Capacity Obligation Quantities (RCOQ) for every facility, from AEMO's monthly CSVs in data/rcoq/.

The files have one row per facility (component) per 30-min trading interval, with the RCOQ for each of
its six 5-min dispatch intervals in "Rcoq DI1" (HH:00 / HH:30) ... "Rcoq DI6" (HH:25 / HH:55).
"Rcoq" is the final value (clause 4.12.4); "Stem Rcoq" is the day-ahead estimate and is not used.
"""

import glob

import pandas as pd

from tools.paths import rcoq_dir

FIELD_NAMES = ["rcoq"]
DI_COLUMNS = [f"Rcoq DI{i}" for i in range(1, 7)]


def load_raw(folder=rcoq_dir) -> pd.DataFrame:
    """All monthly RCOQ files in folder, stacked."""
    columns = ["Facility Code", "Trading Interval"] + DI_COLUMNS
    files = sorted(glob.glob(str(folder / "*.csv")))
    return pd.concat([pd.read_csv(f, usecols=columns) for f in files], ignore_index=True)


def to_5min(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw RCOQ rows -> one MW column per facility on a 5-min dispatch_interval index (Australia/Perth)."""
    long = raw.melt(id_vars=["Facility Code", "Trading Interval"], value_vars=DI_COLUMNS,
                    var_name="di", value_name="rcoq_mw")

    # DI1 starts at the trading interval, DI2 five minutes later, ... DI6 25 minutes later
    di_number = long["di"].str.removeprefix("Rcoq DI").astype(int)
    start = pd.to_datetime(long["Trading Interval"]) + pd.to_timedelta((di_number - 1) * 5, unit="min")
    long["dispatch_interval"] = start.dt.tz_localize("Australia/Perth")

    # a facility can be listed under two component names over time (never at once), so summing is safe
    return long.pivot_table(index="dispatch_interval", columns="Facility Code", values="rcoq_mw", aggfunc="sum")


def build_rcoq_data() -> dict[str, pd.DataFrame]:
    """{"rcoq": 5-min RCOQ for every facility}, in the shape catalog.extract expects."""
    return {"rcoq": to_5min(load_raw())}
