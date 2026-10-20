"""Whole-timeline view of data quality, to justify the analysis period starting at START_DATE.

Top: daily mean fleet SOC. Bottom: % of 5-min SOC readings present,
per battery per month.

    python -m visualisation.plot_data_period
"""

import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import START_DATE, battery_codes
from tools.io import load_clean
from tools.paths import local_plots_dir
from tools.plot_style import save_figure
from visualisation.plot_trend_band import _draw_sse_events


def full_5min_index(soc: pd.DataFrame) -> pd.DataFrame:
    """soc with a row for every 5-min interval, so intervals missing from the file count as NaN."""
    full = pd.date_range(soc.index.min(), soc.index.max(), freq="5min")
    return soc.reindex(full)


def monthly_availability(soc: pd.DataFrame, codes: list[str]) -> pd.DataFrame:
    """% of 5-min readings present per battery per month, counted from each battery's first reading (NaN before)."""
    out = {}
    for code in codes:
        present = soc[code].notna().astype(float)
        first = soc[code].first_valid_index()
        present = present.where(present.index >= first)  # not yet online -> NaN, not 0%
        out[code] = present.resample("MS").mean() * 100
    return pd.DataFrame(out)


def plot_data_period(
    soc: pd.DataFrame,
    events: pd.DataFrame | None = None,
    start_date: str = START_DATE,
    codes: list[str] | None = None,
    categories: list[str] | None = None,
    filename: str = "data_description/data_period_overview.png",
):
    """Two-panel figure: fleet SOC (optionally with SSE events shaded), and monthly data availability per battery.

    soc: clean SOC frame over the whole timeline (load_clean("soc", start_date=None))
    events: from analysis.sse_events.load_sse_events() to shade them (default: no shading)
    start_date: start of the analysis period, marked on both panels
    codes: batteries in the availability panel (default: battery_codes)
    categories: SSE categories to shade on the top panel (default: ["outage"])
    """
    codes = battery_codes if codes is None else codes
    categories = ["outage"] if categories is None else categories
    soc = full_5min_index(soc)
    cut = pd.Timestamp(start_date, tz=soc.index.tz)
    xlim = (soc.index.min(), soc.index.max())

    fig, (ax_soc, ax_avail) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 2]},
    )

    # top: daily mean fleet SOC
    daily_soc = soc["fleet_soc_pct"].resample("D").mean()
    ax_soc.plot(daily_soc.index, daily_soc.values, color="tab:blue", lw=1, label="Fleet SOC (daily mean)")
    event_handles = []
    if events is not None:
        shaded = events[events["category"].isin(categories)]
        event_handles = _draw_sse_events(ax_soc, shaded, xlim)
    ax_soc.set_ylabel("Fleet SOC (%)")
    ax_soc.set_title("Data before and after the analysis period")
    ax_soc.grid(alpha=0.3)

    # bottom: availability heatmap, one row per battery
    avail = monthly_availability(soc, codes)
    month_edges = list(avail.index) + [avail.index[-1] + pd.offsets.MonthBegin(1)]
    cmap = plt.get_cmap("RdYlGn").copy()  # red = few readings, green = complete
    cmap.set_bad("#e6e6e6")  # before a battery's first reading
    mesh = ax_avail.pcolormesh(
        month_edges, range(len(codes) + 1), avail.T.values, cmap=cmap, vmin=0, vmax=100, shading="flat",
    )
    ax_avail.set_yticks([i + 0.5 for i in range(len(codes))])
    ax_avail.set_yticklabels(codes, fontsize=8)
    ax_avail.invert_yaxis()
    ax_avail.set_title("SOC readings available per month (%) - grey: battery not yet online", fontsize=10)
    # colorbar in its own inset outside the panel, so both panels keep the same width
    cax = ax_avail.inset_axes([1.01, 0, 0.015, 1])
    fig.colorbar(mesh, cax=cax, label="% SOC readings available")

    # analysis period start on both panels
    for ax in (ax_soc, ax_avail):
        ax.axvline(cut, color="red", ls="--", lw=1.5)
    ax_soc.text(cut, ax_soc.get_ylim()[1], f"  analysis period from {start_date}", color="red",
                va="top", ha="left", fontsize=9)

    ax_soc.set_xlim(xlim)
    handles = ax_soc.get_legend_handles_labels()[0] + event_handles
    ax_soc.legend(handles=handles, loc="upper left", fontsize=9)

    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)


def main():
    soc = load_clean("soc", start_date=None)
    plot_data_period(soc)


if __name__ == "__main__":
    main()
