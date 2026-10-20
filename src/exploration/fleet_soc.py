import pandas as pd
import matplotlib.pyplot as plt

from tools.io import load_clean
from tools.paths import local_plots_dir
from tools.constants import battery_codes
from tools.plot_style import SOC_BINS, save_figure
from tools.seasons import SEASONS, season_of
from visualisation.plot_distributions import _plot_one_distribution
from visualisation.plot_fleet_capacity import plot_fleet_capacity_evolution
from visualisation.plot_time_of_day import plot_timeofday_box
from visualisation.plot_trend_band import trend_band_event_plot


pd.set_option("display.max_columns", None)

soc_clean = load_clean("soc")  # analysis period only (from START_DATE)
PERIOD = f"{soc_clean.index.min():%b %Y} - {soc_clean.index.max():%b %Y}"  # for plot titles


fleet = soc_clean["fleet_soc_pct"]
# fleet batteries only (no ALINTA_WGP_ESR1), plus the fleet column
pct_cols = [f"{code}_soc_pct" for code in battery_codes] + ["fleet_soc_pct"]

thres = 40
esroi = soc_clean[pct_cols].at_time("17:30")
esroi_low = esroi.where(esroi <= thres)

PAST_YEAR = slice("2025-08-20", "2026-08-20")


def plot_fleet_soc_distribution(soc, title, filename):
    """Histogram + normal fit of a fleet SOC series."""
    fig, ax = plt.subplots(figsize=(10, 6))
    _plot_one_distribution(ax, soc, bicolor=False, bins=SOC_BINS)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Fleet SOC (%)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)


def seasonal_fleet_distribution(soc=None, filename="fleet_description/fleet_soc_distribution_by_season.png"):
    """Fleet SOC distribution, one panel per season. soc: fleet SOC series (default: analysis period)."""
    soc = fleet if soc is None else soc
    period = f"{soc.index.min():%b %Y} - {soc.index.max():%b %Y}"
    seasons = season_of(soc.index)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), squeeze=False)

    for i, s in enumerate(SEASONS):
        ax = axes[i // 2][i % 2]
        mask = seasons == s

        _plot_one_distribution(ax, soc.loc[mask], bicolor=False, bins=SOC_BINS)
        ax.set_xlim(0, 100)

        ax.set_title(s)
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel("Density")

    fig.suptitle(f"Fleet SOC by season ({period})")
    fig.tight_layout()
    save_figure(fig, local_plots_dir / filename)


def pie_chart(filename="batteries_comparison/low_soc_per_battery_pie_chart.png"):
    """Share of low-SOC days at ESROI start, per battery."""
    counts = esroi_low.drop(columns="fleet_soc_pct").count()
    counts.index = counts.index.str.removesuffix("_soc_pct")
    counts = counts.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, _, autotexts = ax.pie(
        counts.values,
        autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",  # hide % on tiny slices
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 1},
        pctdistance=0.75,
    )

    ax.legend(
        wedges,
        [f"{c} ({n})" for c, n in counts.items()],
        title="Battery (low SOC days)",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        frameon=False,
    )
    ax.set_title("Low SOC (below 40%) at ESROI start - Share by battery")
    ax.set_aspect("equal")
    fig.tight_layout()

    save_figure(fig, local_plots_dir / filename)


def main():
    esroi_soc = fleet.at_time("17:30")
    trend_title = "Fleet SOC trend (at ESROI start - 17:30)"

    # fleet_evolution/
    trend_band_event_plot(
        esroi_soc,
        ylabel="Fleet SOC (%)",
        title=f"{trend_title}: monthly median with spread",
        filename="fleet_evolution/fleet_soc_evolution_trend_events.png",
    )
    trend_band_event_plot(
        esroi_soc,
        period="season",
        ylabel="Fleet SOC (%)",
        title=f"{trend_title}: seasonal median with spread",
        filename="fleet_evolution/fleet_soc_evolution_trend_seasonal_events.png",
    )

    # fleet_description/
    plot_fleet_soc_distribution(
        fleet,
        f"Fleet SOC distribution ({PERIOD})",
        "fleet_description/fleet_soc_distribution.png",
    )
    plot_fleet_soc_distribution(
        esroi_soc,
        f"Fleet SOC distribution at ESROI start (17:30) ({PERIOD})",
        "fleet_description/fleet_soc_distribution_esroi_start.png",
    )
    plot_fleet_soc_distribution(
        fleet.loc[PAST_YEAR],
        "Fleet SOC distribution in the past year (Aug 2025 - Aug 2026)",
        "fleet_description/fleet_soc_distribution_past_year.png",
    )
    plot_fleet_soc_distribution(
        fleet.loc[PAST_YEAR].at_time("17:30"),
        "Fleet SOC distribution in the past year (Aug 2025 - Aug 2026) at ESROI start (17:30)",
        "fleet_description/fleet_soc_distribution_past_year_esroi_start.png",
    )
    seasonal_fleet_distribution()
    seasonal_fleet_distribution(
        load_clean("soc", start_date=None)["fleet_soc_pct"],
        filename="fleet_description/fleet_soc_distribution_by_season_full_history.png",
    )
    # capacity chart shows all capacity built so far, so it uses the whole timeline
    plot_fleet_capacity_evolution(load_clean("soc", start_date=None), filename="fleet_description/fleet_capacity_evolution.png")

    # summary_plots/
    plot_timeofday_box(
        fleet,
        "Fleet SOC (%)",
        "Fleet SOC by time of day",
        "summary_plots/fleet_soc_timeofday_box_grid.png",
    )

    # batteries_comparison/
    pie_chart()


if __name__ == "__main__":
    main()
