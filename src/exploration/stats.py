from matplotlib.pylab import percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from tools.paths import extracted_data_dir, clean_data_dir, local_plots_dir
from tools.constants import battery_capacity_MW, battery_capacity_MWh, battery_codes, codes_in
from tools.plot_style import UNIT_COLORS, save_figure
from visualisation.plot_distributions import plot_distribution_grid, _plot_one_distribution
from visualisation.plot_fleet_capacity import plot_fleet_capacity_evolution


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




def monthly_band_plot(df, col=None, date_col=None, min_days=20,
                      show_mean=True, title=None):
    """
    Monthly median line with a shaded p10-p90 band.

    df        : DataFrame with daily values
    col       : name of the value column (defaults to the only/first numeric column)
    date_col  : name of the date column, if dates aren't already the index
    min_days  : months with fewer readings than this are flagged as unreliable
    show_mean : also draw the monthly mean as a dashed line
    """
    # accept a Series as well as a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name if df.name is not None else 'value')

    d = df.copy()

    # get a DatetimeIndex
    if date_col is not None:
        d[date_col] = pd.to_datetime(d[date_col])
        d = d.set_index(date_col)
    elif not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    d = d.sort_index()

    if col is None:
        col = d.select_dtypes('number').columns[0]
    s = d[col]

    # monthly summary
    g = s.resample('MS')
    m = pd.DataFrame({
        'median': g.median(),
        'mean':   g.mean(),
        'p10':    g.quantile(0.10),
        'p25':    g.quantile(0.25),
        'p75':    g.quantile(0.75),
        'p90':    g.quantile(0.90),
        'n':      g.count(),
    })
    m['low_n'] = m['n'] < min_days

    # plot
    fig, ax = plt.subplots(figsize=(11, 5))

    ax.fill_between(m.index, m['p10'], m['p90'],
                    alpha=0.18, color='C0', label='p10–p90 (most days)')
    ax.fill_between(m.index, m['p25'], m['p75'],
                    alpha=0.30, color='C0', label='p25–p75 (middle half)')
    ax.plot(m.index, m['median'], color='C0', lw=2, marker='o', ms=4,
            label='Monthly median')
    if show_mean:
        ax.plot(m.index, m['mean'], color='C1', lw=1.2, ls='--',
                label='Monthly mean')

    # flag months with too few readings
    bad = m[m['low_n']]
    if len(bad):
        ax.scatter(bad.index, bad['median'], s=90, facecolors='none',
                   edgecolors='red', lw=1.5, zorder=5,
                   label=f'Fewer than {min_days} days of data')
        
    seasons = {12: 'Summer', 3: 'Autumn', 6: 'Winter', 9: 'Spring'}

    def season_label(x, pos):
        dt = mdates.num2date(x)
        return f"{seasons[dt.month]}\n{dt:%b %Y}"

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_major_formatter(FuncFormatter(season_label))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.set_ylabel("Fleet SOC (%)")
    ax.set_title(title or f'Fleet SOC trend (at ESROI start - 17:30): monthly median with spread')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)
    fig.tight_layout()
    save_figure(fig, local_plots_dir / "fleet_soc_evolution_trend.png")

    return fig, ax, m



# fig, ax, summary = monthly_band_plot(fleet.at_time("17:30"))
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



def seasonal_fleet():
    season = {
        "summer": [12, 1, 2],
        "autumn": [3, 4, 5],
        "winter": [6, 7, 8],
        "spring": [9, 10, 11]
    }
    
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 6), squeeze=False)
    
    for i, s in enumerate(season.keys()):
        ax = axes[i // 2][i % 2]
        col_name = s.capitalize()
        mask = fleet.index.month.isin(season[s])

        _plot_one_distribution(ax, fleet.loc[mask], bicolor=False)
        
        ax.set_title(col_name)
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

