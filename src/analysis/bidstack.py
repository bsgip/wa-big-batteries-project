"""Battery bid stacks as MW$ (quantity x submitted price): loading, gridding and rolling change."""

import pandas as pd

from tools.constants import battery_codes
from tools.paths import extracted_data_dir, local_data_dir

BIDSTACK_CSV = local_data_dir / "battery_bidstack.csv"


def extract_battery_bidstack():
    """Save the battery rows of extracted/bidstack.parquet to BIDSTACK_CSV."""
    filename = extracted_data_dir / "bidstack.parquet"
    df = pd.read_parquet(filename)
    df = df.loc[df["code"].isin(battery_codes)]
    cols = ["dispatch_interval", "code", "tranche", "quantity", "submitted_price"]
    df = df[cols]

    try:
        df.to_csv(BIDSTACK_CSV, index=False)
        print("succesfully saved to csv")
    except Exception as e:
        print(e)


def load_bidstack() -> pd.DataFrame:
    """Load BIDSTACK_CSV and add a mw_dollars column (quantity x submitted_price).

    Returns:
        One row per (dispatch_interval, code, tranche).
    """
    df = pd.read_csv(BIDSTACK_CSV, index_col=0, parse_dates=["dispatch_interval"])
    df["mw_dollars"] = df["quantity"] * df["submitted_price"]
    return df


def trim_to_window(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    """Keep the rows of df between start and end, in the frame's own timezone.

    Args:
        df: Bid stack with a dispatch_interval column.
        start: First timestamp to keep, e.g. "2026-01-10"; None for no lower bound.
        end: Last timestamp to keep; None for no upper bound.

    Returns:
        The trimmed frame (df itself if both bounds are None).
    """
    if start is None and end is None:
        return df
    tz = df["dispatch_interval"].dt.tz
    bounds = slice(
        pd.Timestamp(start, tz=tz) if start else None,
        pd.Timestamp(end, tz=tz) if end else None,
    )
    return df.set_index("dispatch_interval").sort_index().loc[bounds].reset_index()


def rolling_change(
    df: pd.DataFrame,
    window: str = "1D",
    signed: bool = False,
    freq: str = "5min",
    min_frac: float = 0.5,
) -> pd.DataFrame:
    """Interval-to-interval change in MW$ per unit and tranche, as a rolling mean.

    The frame is reindexed onto a continuous `freq` grid first, so a data gap gives NaN
    rather than one large change across it.

    Args:
        df: Bid stack from load_bidstack.
        window: Rolling window, e.g. "1D".
        signed: If False (default), average the absolute change; if True, the signed change.
        freq: Grid spacing of the data.
        min_frac: Minimum share of the window that must have values.

    Returns:
        Frame indexed by interval, with (code, tranche) columns.
    """
    wide = df.pivot_table(index="dispatch_interval", columns=["code", "tranche"], values="mw_dollars", aggfunc="mean")
    print(wide)
    grid = pd.date_range(wide.index.min(), wide.index.max(), freq=freq, tz=wide.index.tz)
    wide = wide.reindex(grid)

    change = wide.diff()
    if not signed:
        change = change.abs()

    periods = max(1, int(pd.Timedelta(window) / pd.Timedelta(freq)))
    return change.rolling(window, min_periods=max(1, int(periods * min_frac))).mean()


def tranche_grid(
    df: pd.DataFrame,
    freq: str | None = "D",
    agg: str = "mean",
    start: str | None = None,
    end: str | None = None,
    values: str = "level",
) -> dict[str, pd.DataFrame]:
    """Build a tranche x time grid of MW$ for each unit.

    Args:
        df: Bid stack from load_bidstack.
        freq: Resampling frequency, e.g. "D"; None keeps every dispatch interval.
        agg: How to aggregate within each period, e.g. "mean".
        start: First timestamp to keep; None for no lower bound.
        end: Last timestamp to keep; None for no upper bound.
        values: "level" for MW$, or "change" for the difference from the previous period
            (NaN across gaps).

    Returns:
        {code: frame indexed by period, one column per tranche}.
    """
    df = trim_to_window(df, start, end)
    index = pd.Grouper(key="dispatch_interval", freq=freq) if freq else "dispatch_interval"

    grids = {}
    for code in sorted(df["code"].unique()):
        wide = (
            df[df["code"] == code]
            .pivot_table(index=index, columns="tranche", values="mw_dollars", aggfunc=agg)
            .sort_index()
        )
        if values == "change":
            step = pd.Timedelta(freq) if freq else pd.Timedelta("5min")
            wide = wide.reindex(pd.date_range(wide.index.min(), wide.index.max(), freq=step, tz=wide.index.tz))
            wide = wide.diff()
        grids[code] = wide
    return grids
