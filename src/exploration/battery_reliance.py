from analysis.energy import (
    credit_equivalent_soc_mwh,
    fleet_net_energy_mwh,
    fleet_net_energy_pct,
    fleet_required_energy_mwh,
    monthly_share_below,
    typical_credit_equivalent_mwh,
    window_energy_mwh,
)
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, battery_capacity_MWh, battery_codes, commissioning_dates
from tools.io import load_clean
from tools.paths import local_plots_dir
from tools.plot_style import DISCHARGE_COLOR, save_figure
from visualisation.plot_soc_vs_credit import (
    plot_share_below_by_month,
    plot_below_credit_summary,
    plot_soc_vs_credit_grid,
)
from visualisation.plot_trend_band import (
    mark_commissioning,
    mark_period_medians,
    stacked_median_plot,
    trend_band_plot,
)

MIDDAY_START = "09:00"
MIDDAY_END = "15:00"

# a battery is "low" at ESROI start below this SOC (% of its own capacity)
LOW_SOC_PCT = 50


def main():
    soc = load_clean("soc")
    power = load_clean("power")
    demand = load_clean("demand")["demand"]
    commissioning = commissioning_dates(soc, power)

    # Fleet utilisation
    fleet_utlisation = fleet_net_energy_pct(power, commissioning, PEAK_ESROI_START, PEAK_ESROI_END)

    # Fleet reliance
    fleet_mwh = fleet_net_energy_mwh(power, PEAK_ESROI_START, PEAK_ESROI_END)
    demand_mwh = window_energy_mwh(demand, PEAK_ESROI_START, PEAK_ESROI_END)

    fleet_reliance = fleet_mwh / demand_mwh * 100

    # Fleet coverage
    stored_mwh_start = soc["fleet_soc_MWh"].at_time(PEAK_ESROI_START)
    stored_mwh_start.index = stored_mwh_start.index.normalize()
    fleet_coverage = stored_mwh_start / demand_mwh * 100

    stored_pct_end = soc["fleet_soc_pct"].at_time(PEAK_ESROI_END)

    # Obligation headroom: stored energy at ESROI start vs energy needed to meet the RCOQ through ESROI
    rcoq = load_clean("rcoq")
    required_mwh = fleet_required_energy_mwh(rcoq, PEAK_ESROI_START, PEAK_ESROI_END)

    # dropna: RCOQ data runs past the end of the SOC data
    headroom_mwh = (stored_mwh_start - required_mwh).dropna()
    headroom_pct = (headroom_mwh / demand_mwh * 100).dropna()

    print(stored_pct_end)

    # trend_band_plot(
    #     df=fleet_utlisation,
    #     ylabel="Fleet utilisation (%)\nFleet net energy (MWh) / Fleet total capacity (MWh)",
    #     title=f"Fleet utilisation (%) during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
    #     filename="fleet_evolution/fleet_utilisation_evolution_trend.png",
    # )

    trend_band_plot(
        df=fleet_reliance,
        ylabel="Fleet reliance (%)\nFleet net energy (MWh) / Net demand (MWh)",
        title=f"Fleet reliance (%) during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename="fleet_evolution/fleet_reliance_evolution_trend.png",
    )

    # full history (from Sep 2023), to match the price trend plots - not just the analysis period
    trend_band_plot(
        df=load_clean("demand", start_date=None)["demand"],
        ylabel="Operational demand (MW)",
        title="Operational demand evolution",
        filename="demand_evolution_trend.png",
    )

    # DPV is only clean from 2024-11-07 (see data_processing/dpv.py) - start there, so the
    # deliberately-blanked months before it aren't drawn as months with too little data
    dpv = load_clean("dpv", start_date=None)["dpv"]
    trend_band_plot(
        df=dpv.loc[dpv.first_valid_index():],
        ylabel="Rooftop PV, DPV (MW)",
        title="Distributed PV evolution",
        filename="dpv_evolution_trend.png",
    )

    # trend_band_event_plot(
    #     df=fleet_coverage,
    #     ylabel="Fleet energy coverage (%)\nStored energy at ESROI start (MWh) / Net demand (MWh)",
    #     title=f"Fleet coverage (%) of demand during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
    #     filename="fleet_coverage_evolution_trend.png",
    # )

    # trend_band_plot(
    #     df=headroom_mwh,
    #     ylabel="Obligation headroom (MWh)\nStored energy at ESROI start (MWh) - RCOQ energy (MWh)",
    #     title=f"Fleet headroom over its RCOQ during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
    #     filename="fleet_evolution/fleet_obligation_headroom_trend.png",
    # )

    headroom_pct_file = "fleet_evolution/fleet_obligation_headroom_pct_trend.png"
    fig, ax, _ = trend_band_plot(
        df=headroom_pct,
        ylabel="Obligation headroom (% of ESROI demand)",
        title=f"Fleet headroom over its RCOQ as a share of ESROI demand ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename=headroom_pct_file,
    )
    # formula as a note on the plot instead of in the axis label, then save again over the same file
    ax.text(
        0.01,
        0.03,
        "Headroom = (stored energy at ESROI start − RCOQ energy) / ESROI demand × 100",
        transform=ax.transAxes,
        fontsize=8.5,
        va="bottom",
        ha="left",
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    save_figure(fig, local_plots_dir / headroom_pct_file)

    # trend_band_event_plot(
    #     df=stored_pct_end,
    #     ylabel="Fleet SOC (%)",
    #     title=f"Fleet SOC (%) at the end of ESROI ({PEAK_ESROI_END})",
    #     filename="fleet_soc_end_esroi_evolution_trend.png",
    # )

    # ESROI demand vs fleet reliance: monthly medians on one time axis
    demand_vs_reliance_file = "fleet_evolution/esroi_demand_vs_reliance_monthly_median.png"
    fig, axes, _ = stacked_median_plot(
        panels=[
            (demand_mwh, "ESROI demand (MWh)", "C0"),
            (fleet_reliance, "Fleet reliance (%)\nFleet net energy / demand", DISCHARGE_COLOR),
        ],
        title=f"Monthly median of ESROI demand and fleet reliance ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename=demand_vs_reliance_file,
    )
    # year-on-year change: median over the same months a year apart (Dec-Sep, the full months both years
    # have), so seasonality doesn't read as change; then save again over the same file
    same_months = [("2024-12-01", "2025-09-30"), ("2025-12-01", "2026-09-30")]
    mark_period_medians(axes[0], demand_mwh, same_months, value_format="{:,.0f} MWh")
    mark_period_medians(axes[1], fleet_reliance, same_months, value_format="{:.1f}%")
    save_figure(fig, local_plots_dir / demand_vs_reliance_file)

    # Fleet utilisation: monthly median, with when each battery was commissioned
    utilisation_file = "fleet_evolution/fleet_utilisation_monthly_median.png"
    fig, axes, _ = stacked_median_plot(
        panels=[
            (fleet_utlisation, "Fleet utilisation (%)\nFleet net energy / fleet capacity", DISCHARGE_COLOR),
        ],
        title=f"Monthly median of fleet utilisation during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename=utilisation_file,
    )
    # new capacity lowers utilisation until the unit is in regular use, so mark when each arrived,
    # then save again over the same file
    # only the batteries utilisation is calculated over (battery_codes, so no ALINTA_WGP_ESR1)
    fleet_commissioning = {code: date for code, date in commissioning.items() if code in battery_codes}
    mark_commissioning(axes[0], fleet_commissioning, battery_capacity_MWh)
    # same year-on-year comparison as demand vs reliance; legend at the bottom, clear of the commissioning labels
    mark_period_medians(axes[0], fleet_utlisation, same_months, value_format="{:.1f}%", legend_loc="lower right")
    save_figure(fig, local_plots_dir / utilisation_file)

    # Per battery: SOC at ESROI start vs the charge its RCOQ needs for the whole window
    soc_start = soc[battery_codes].at_time(PEAK_ESROI_START)
    soc_start.index = soc_start.index.normalize()
    credit_soc = credit_equivalent_soc_mwh(rcoq, PEAK_ESROI_START, PEAK_ESROI_END, battery_codes)
    plot_soc_vs_credit_grid(
        soc_start,
        credit_soc,
        battery_capacity_MWh,
        title=f"SOC at ESROI start ({PEAK_ESROI_START}) vs credit-equivalent SOC, per battery",
        filename="soc_at_esroi_start_vs_credit_per_battery.png",
    )

    # Per battery: how often / how far SOC at ESROI start is below ONE credit number,
    # its usual level in the 2025-26 capacity year - every day since its first SOC
    # reading counts, not just days its RCOQ had started
    credit_mwh = typical_credit_equivalent_mwh(credit_soc, "2025-10-01", "2026-09-30")
    _, _, below_credit = plot_below_credit_summary(
        soc_start,
        credit_mwh,
        title=f"SOC at ESROI start ({PEAK_ESROI_START}) below credit equivalent, per battery",
        filename="batteries_comparison/soc_below_credit_summary.png",
    )
    print(below_credit.round(1))

    # Low charge by month, against the size of the fleet being counted. "Low" is each battery's own
    # SOC % below LOW_SOC_PCT - not its credit equivalent, which is 88-100% of capacity for some
    # batteries and so makes almost every day "low"
    soc_start_pct = soc[[f"{code}_soc_pct" for code in battery_codes]].at_time(PEAK_ESROI_START)
    soc_start_pct.index = soc_start_pct.index.normalize()
    soc_start_pct.columns = battery_codes
    share_low = monthly_share_below(soc_start_pct, LOW_SOC_PCT)
    plot_share_below_by_month(
        soc_start_pct,
        share_low,
        battery_capacity_MWh,
        ylabel=f"Days with SOC below {LOW_SOC_PCT}% (%)",
        title=f"Days with SOC at ESROI start ({PEAK_ESROI_START}) below {LOW_SOC_PCT}%, by month",
        filename=f"batteries_comparison/soc_below_{LOW_SOC_PCT}pct_by_month.png",
    )


if __name__ == "__main__":
    main()
