from matplotlib.pylab import percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tools.paths import extracted_data_dir, clean_data_dir, local_plots_dir
from tools.constants import battery_capacity_MW, battery_capacity_MWh, battery_codes, codes_in
from tools.plot_style import UNIT_COLORS, save_figure
from tools.seasons import SEASONS, season_of
from visualisation.plot_distributions import plot_distribution_grid, _plot_one_distribution
from visualisation.plot_fleet_capacity import plot_fleet_capacity_evolution
from visualisation.plot_trend_band import trend_band_plot, trend_band_event_plot


pd.set_option("display.max_columns", None)

soc_ext_path = extracted_data_dir / "soc.parquet"
soc_clean_path = clean_data_dir / "soc.parquet"

soc_ext = pd.read_parquet(soc_ext_path)
soc_clean = pd.read_parquet(soc_clean_path)


fleet = soc_clean["fleet_soc_pct"]
pct_cols = [col for col in soc_clean.columns if "soc_pct" in col.lower()]

thres = 40
esroi = soc_clean[pct_cols].at_time("17:30")
esroi_low = esroi.where(esroi <= thres)





soc_labels = dict(ylabel="Fleet SOC (%)", title="Fleet SOC trend (at ESROI start - 17:30)")
# fig, ax, summary = trend_band_plot(fleet.at_time("17:30"), **soc_labels, filename="fleet_soc_evolution_trend.png")
# fig, ax, summary = trend_band_plot(fleet.at_time("17:30"), period="season", show_sse=False, **soc_labels, filename="fleet_soc_evolution_trend_seasonal.png")
fig, axes, summary = trend_band_event_plot(fleet.at_time("17:30"), **soc_labels, filename="fleet_soc_evolution_trend_events.png")
# fig, axes, summary = trend_band_event_plot(fleet.at_time("17:30"), period="season", **soc_labels, filename="fleet_soc_evolution_trend_seasonal_events.png")
# plt.show()
# summary.round(2)







def fleet_soc():
    past_year = fleet.loc["2025-08-20":"2026-08-20"]
    past_esroi = past_year.at_time("17:30")
    fig, ax = plt.subplots(figsize=(10,6))
    _plot_one_distribution(ax, past_esroi, bicolor=False)
    ax.set_xlabel("Fleet SOC (%)")
    ax.set_ylabel("Density")
    ax.set_title("Fleet SOC distribution in the past year (Aug 2025 - Aug 2026) at ESROI start (17:30)")
    fig.tight_layout()
    save_figure(fig, local_plots_dir / "fleet_soc_distribution_past_year_esroi_start.png")
    plt.show()


fleet_soc()



def seasonal_fleet_distribution():
    seasons = season_of(fleet.index)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), squeeze=False)
    
    for i, s in enumerate(SEASONS):
        ax = axes[i // 2][i % 2]
        mask = seasons == s

        _plot_one_distribution(ax, fleet.loc[mask], bicolor=False)
        
        ax.set_title(s)
        ax.set_xlabel("SOC (%)")
        ax.set_ylabel("Density")
    
    fig.suptitle("Fleet SOC by season")
    fig.tight_layout()
    # save_figure(fig, local_plots_dir / "fleet_soc_distribution_by_season.png")
    
    plt.show()



def pie_chart():
    counts = esroi_low.drop(columns="fleet_soc_pct").count()
    counts.index = counts.index.str.removesuffix("_soc_pct")
    counts = counts.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    wedges, _, autotexts = ax.pie(
        counts.values,
        autopct=lambda p: f"{p:.0f}%" if p >= 4 else "",   # hide % on tiny slices
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
    
    save_figure(fig, local_plots_dir / "low_soc_per_battery_pie_chart.png")
    plt.show()

