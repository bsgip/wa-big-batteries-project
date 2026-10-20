from analysis.energy import fleet_net_energy_pct, fleet_net_energy_mwh, fleet_required_energy_mwh, window_energy_mwh
from tools.io import load_clean
from tools.constants import commissioning_dates, PEAK_ESROI_START, PEAK_ESROI_END
from tools.paths import local_plots_dir
from tools.plot_style import save_figure
from visualisation.plot_trend_band import trend_band_event_plot

MIDDAY_START = "09:00"
MIDDAY_END = "15:00"


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

    trend_band_event_plot(
        df=fleet_utlisation,
        ylabel="Fleet utilisation (%)\nFleet net energy (MWh) / Fleet total capacity (MWh)",
        title=f"Fleet utilisation (%) during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename="fleet_evolution/fleet_utilisation_evolution_trend.png"
    )
    
    trend_band_event_plot(
        df=fleet_reliance,
        ylabel="Fleet reliance (%)\nFleet net energy (MWh) / Net demand (MWh)",
        title=f"Fleet reliance (%) during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename="fleet_evolution/fleet_reliance_evolution_trend.png"
    )
    
    trend_band_event_plot(
        df=demand,
        ylabel="Operational demand (MWh)",
        title="Operational demand evolution",
        filename="demand_evolution_trend.png"
    )
    
    trend_band_event_plot(
        df=fleet_coverage,
        ylabel="Fleet energy coverage (%)\nStored energy at ESROI start (MWh) / Net demand (MWh)",
        title=f"Fleet coverage (%) of demand during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename="fleet_coverage_evolution_trend.png"
    )
    
    trend_band_event_plot(
        df=headroom_mwh,
        ylabel="Obligation headroom (MWh)\nStored energy at ESROI start (MWh) - RCOQ energy (MWh)",
        title=f"Fleet headroom over its RCOQ during ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename="fleet_evolution/fleet_obligation_headroom_trend.png"
    )
    
    headroom_pct_file = "fleet_evolution/fleet_obligation_headroom_pct_trend.png"
    fig, (ax, _), _ = trend_band_event_plot(
        df=headroom_pct,
        ylabel="Obligation headroom (% of ESROI demand)",
        title=f"Fleet headroom over its RCOQ as a share of ESROI demand ({PEAK_ESROI_START}-{PEAK_ESROI_END})",
        filename=headroom_pct_file
    )
    # formula as a note on the plot instead of in the axis label, then save again over the same file
    ax.text(
        0.01, 0.03,
        "Headroom = (stored energy at ESROI start − RCOQ energy) / ESROI demand × 100",
        transform=ax.transAxes, fontsize=8.5, va="bottom", ha="left",
        bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    save_figure(fig, local_plots_dir / headroom_pct_file)

    trend_band_event_plot(
        df=stored_pct_end,
        ylabel="Fleet SOC (%)",
        title=f"Fleet SOC (%) at the end of ESROI ({PEAK_ESROI_END})",
        filename="fleet_soc_end_esroi_evolution_trend.png"
    )


if __name__ == "__main__":
    main()