import pandas as pd

from analysis.energy import fleet_soc_drop_pct, low_soc_days_per_battery
from tools.io import load_clean
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START
from visualisation.plot_distributions import plot_fleet_soc_distribution, plot_seasonal_fleet_soc_distribution
from visualisation.plot_fleet_capacity import plot_fleet_capacity_evolution
from visualisation.plot_pie import plot_share_pie
from visualisation.plot_time_of_day import plot_timeofday_box
from visualisation.plot_trend_band import period_box_plot, trend_band_plot


pd.set_option("display.max_columns", None)

PAST_YEAR = slice("2025-08-20", "2026-08-20")
LOW_SOC_PCT = 40


def main():
    soc_clean = load_clean("soc")  # analysis period only (from START_DATE)
    period = f"{soc_clean.index.min():%b %Y} - {soc_clean.index.max():%b %Y}"  # for plot titles
    fleet = soc_clean["fleet_soc_pct"]

    esroi_soc = fleet.at_time("17:30")
    trend_title = "Fleet SOC trend (at ESROI start - 17:30)"

    # fleet_evolution/
    trend_band_plot(
        esroi_soc,
        ylabel="Fleet SOC (%)",
        title=f"{trend_title}: monthly median with spread",
        filename="fleet_evolution/fleet_soc_evolution_trend.png",
    )
    trend_band_plot(
        esroi_soc,
        period="season",
        ylabel="Fleet SOC (%)",
        title=f"{trend_title}: seasonal median with spread",
        filename="fleet_evolution/fleet_soc_evolution_trend_seasonal.png",
    )
    trend_band_plot(
        fleet.at_time(PEAK_ESROI_END),
        ylabel="Fleet SOC (%)",
        title=f"Fleet SOC trend (at ESROI end - {PEAK_ESROI_END}): monthly median with spread",
        filename="fleet_evolution/fleet_soc_esroi_end_trend.png",
    )
    period_box_plot(
        esroi_soc,
        ylim=(0, 100),
        ylabel="Fleet SOC (%)",
        title="Fleet SOC at ESROI start (17:30), by month",
        filename="fleet_evolution/fleet_soc_esroi_start_monthly_box.png",
    )
    # how much of that charge is used: SOC at ESROI start minus at ESROI end
    soc_used = fleet_soc_drop_pct(soc_clean, PEAK_ESROI_START, PEAK_ESROI_END)
    soc_used_label = f"Fleet SOC used (%)\nSOC at {PEAK_ESROI_START} − SOC at {PEAK_ESROI_END}"
    period_box_plot(
        soc_used,
        ylabel=soc_used_label,
        title=f"Fleet SOC used during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END}), by month",
        filename="fleet_evolution/fleet_soc_used_esroi_monthly_box.png",
    )
    trend_band_plot(
        soc_used,
        ylabel=soc_used_label,
        title=f"Fleet SOC used during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END}): monthly median with spread",
        filename="fleet_evolution/fleet_soc_used_esroi_trend.png",
    )

    # fleet_description/
    plot_fleet_soc_distribution(
        fleet,
        f"Fleet SOC distribution ({period})",
        "fleet_description/fleet_soc_distribution.png",
    )
    plot_fleet_soc_distribution(
        esroi_soc,
        f"Fleet SOC distribution at ESROI start (17:30) ({period})",
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
    plot_seasonal_fleet_soc_distribution(
        fleet,
        filename="fleet_description/fleet_soc_distribution_by_season.png",
    )
    plot_seasonal_fleet_soc_distribution(
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
    plot_share_pie(
        low_soc_days_per_battery(soc_clean, PEAK_ESROI_START, LOW_SOC_PCT),
        legend_title="Battery (low SOC days)",
        title=f"Low SOC (below {LOW_SOC_PCT}%) at ESROI start - Share by battery",
        filename="batteries_comparison/low_soc_per_battery_pie_chart.png",
    )


if __name__ == "__main__":
    main()
