import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.paths import local_plots_dir
from tools.constants import battery_capacity_MWh, battery_codes, codes_in
from tools.plot_style import UNIT_COLORS, save_figure


def semester_labels(index: pd.DatetimeIndex) -> pd.Index:
    """"YYYY-S1" / "YYYY-S2" label for each timestamp in index."""
    semester = np.where(index.month <= 6, "S1", "S2")
    return pd.Index(index.year.astype(str) + "-" + semester, name="semester")


def fleet_capacity_by_semester(soc_df: pd.DataFrame, codes: list[str] | None = None) -> pd.DataFrame:
    """Rated capacity of each battery per semester.

    Args:
        soc_df: Clean SOC frame (per-battery MWh columns).
        codes: Batteries to include; defaults to every battery in soc_df, ALINTA_WGP_ESR1 included.

    Returns:
        One row per semester, one MWh column per battery (0 before its first reading).
    """
    periods = semester_labels(soc_df.index)
    codes_present = codes if codes is not None else codes_in(soc_df)

    commissioned = soc_df[codes_present].notna().cummax()
    contribution = commissioned.mul([battery_capacity_MWh[code] for code in codes_present])

    table = contribution.groupby(periods).max()
    table = table.reindex(sorted(table.index)).fillna(0)
 
    # Only meaningful when the split covers the same fleet the stored total
    # does: a deliberately narrowed `codes` is expected to sum to less, so
    # checking it there would fail on correct output.
    if codes_present == codes_in(soc_df, battery_codes):
        stored_total = soc_df["fleet_capacity_MWh"].groupby(periods).max().reindex(table.index)
        if not np.allclose(table.sum(axis=1), stored_total):
            raise ValueError(
                "per-battery capacity split doesn't sum to fleet_capacity_MWh - "
                "the commissioning rule in data_processing/soc.py has changed"
            )

    return table


def plot_fleet_capacity_evolution(
    soc_df: pd.DataFrame,
    filename: str = "fleet_capacity_evolution.png",
    codes: list[str] | None = None,
):
    """Plot stacked bars of fleet capacity per semester, one segment per battery, total above each bar.

    Args:
        soc_df: Clean SOC frame over the whole timeline.
        filename: Path under local_plots_dir to save the figure to.
        codes: Batteries to include; see fleet_capacity_by_semester.
    """
    table = fleet_capacity_by_semester(soc_df, codes)

    fig, ax = plt.subplots(figsize=(10, 6))

    bottom = np.zeros(len(table))
    # iterate the table's own columns, not a constant: the two must be the
    # same fleet or the segments won't add up to the annotated total
    for code in table.columns:
        values = table[code].to_numpy()
        ax.bar(table.index, values, bottom=bottom, label=code, color=UNIT_COLORS[code], width=0.7)
        bottom += values

    for x, total in enumerate(bottom):
        ax.annotate(
            f"{total:,.0f}", (x, total), textcoords="offset points", xytext=(0, 4),
            ha="center", fontsize="small",
        )

    ax.set_xlabel("Semester")
    ax.set_ylabel("Rated capacity of commissioned fleet (MWh)")
    ax.set_title("Total battery fleet capacity by semester")
    ax.margins(y=0.12)  # headroom for the total labels
    ax.legend(title="Battery", fontsize="small", loc="upper left")

    save_figure(fig, local_plots_dir / filename)


