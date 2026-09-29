from matplotlib.pylab import percentile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch

from tools.paths import extracted_data_dir, clean_data_dir, local_data_dir, local_plots_dir
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




# one colour per SSE category, kept away from the blue/orange the SOC series
# already uses so an event stripe never reads as part of the data
SSE_CATEGORY_COLORS = {
    "heatwave":    "#D7191C",   # red
    "temperature": "#00A6A6",   # teal
    "outage":      "#1A1A1A",   # near-black
    "multiple":    "#7D3AC1",   # violet
    "other":       "#4CAF50",   # green
}

sse_events_path = local_data_dir / "sse_events.csv"


def load_sse_events(path=sse_events_path):
    """SSE events as start/end timestamps plus category.

    Events with no recorded end are treated as ending when they started, i.e.
    drawn as a single line rather than a span.
    """
    ev = pd.read_csv(path, index_col=0)

    start = pd.to_datetime(ev["start_date"] + " " + ev["start_time"].fillna("00:00:00"))
    end = pd.to_datetime(
        ev["end_date"].fillna(ev["start_date"]) + " " + ev["end_time"].fillna("23:59:59")
    )
    end = end.mask(ev["end_date"].isna(), start)

    return pd.DataFrame({"start": start, "end": end, "category": ev["category"]})


def _draw_sse_events(ax, events, xlim):
    """Shade each SSE event on ax: a span if it runs over more than a day,
    a single line if not. Returns legend handles, one per category present."""
    lo, hi = xlim

    # the SOC series is tz-aware (Australia/Perth); the event times in the csv
    # are bare local times, so put them on the same clock before comparing
    if lo.tz is not None:
        events = events.copy()
        for c in ("start", "end"):
            events[c] = events[c].dt.tz_localize(lo.tz)

    drawn = set()
    for _, e in events.iterrows():
        if e["end"] < lo or e["start"] > hi:   # outside the plotted window
            continue
        color = SSE_CATEGORY_COLORS.get(e["category"], "#333333")
        if (e["end"] - e["start"]) >= pd.Timedelta(days=1):
            ax.axvspan(e["start"], e["end"], color=color, alpha=0.15, lw=0, zorder=0)
        else:
            ax.axvline(e["start"], color=color, alpha=0.40, lw=0.9, zorder=0)
        drawn.add(e["category"])

    order = [c for c in SSE_CATEGORY_COLORS if c in drawn]
    return [
        Patch(facecolor=SSE_CATEGORY_COLORS[c], alpha=0.5, label=c.capitalize())
        for c in order
    ]


# How the value column gets grouped before the band is drawn. "season" uses
# quarters starting in December, so the four buckets are exactly the southern
# seasons the x-axis already labels: Dec-Feb summer, Mar-May autumn,
# Jun-Aug winter, Sep-Nov spring.
_PERIOD_SPECS = {
    'month': {
        'freq':  'MS',
        'label': 'Monthly',
        'end':   pd.offsets.MonthEnd(0),
        'stem':  'fleet_soc_evolution_trend',
    },
    'season': {
        'freq':  'QS-DEC',
        'label': 'Seasonal',
        'end':   pd.offsets.QuarterEnd(startingMonth=2),
        'stem':  'fleet_soc_evolution_trend_seasonal',
    },
}


def _prepare_series(df, col=None, date_col=None):
    """One sorted, datetime-indexed Series out of whatever was passed in."""
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
    return d[col]


def _period_summary(s, freq):
    """Per-period median/mean/percentiles, plus the days that fall outside
    their own period's 1.5xIQR fences. Returns (summary frame, outlier days)."""
    g = s.resample(freq)
    m = pd.DataFrame({
        'median': g.median(),
        'mean':   g.mean(),
        'p10':    g.quantile(0.10),
        'p25':    g.quantile(0.25),
        'p75':    g.quantile(0.75),
        'p90':    g.quantile(0.90),
        'min':    g.min(),
        'max':    g.max(),
        'n':      g.count(),
    })

    # Tukey fences off the period's own quartiles: a day is an outlier relative
    # to the period it sits in, so a quiet winter and a volatile summer each
    # get judged on their own spread.
    iqr = m['p75'] - m['p25']
    m['lower_fence'] = m['p25'] - 1.5 * iqr
    m['upper_fence'] = m['p75'] + 1.5 * iqr

    # which row of m each day belongs to - m.index holds the period starts,
    # so the last start at or before a timestamp is that day's period
    pos = m.index.searchsorted(s.index, side='right') - 1
    fences = m.iloc[pos]
    is_outlier = (s.values < fences['lower_fence'].values) | (s.values > fences['upper_fence'].values)
    outliers = s[is_outlier]
    m['n_outliers'] = outliers.resample(freq).count().reindex(m.index).fillna(0).astype(int)

    return m, outliers


def _draw_soc_band(ax, m, outliers, unit, show_mean=True, show_outliers=True):
    """The SOC series itself: p10-p90 and p25-p75 bands, median, mean, outliers."""
    ax.fill_between(m.index, m['p10'], m['p90'],
                    alpha=0.18, color='C0', label='p10–p90 (most days)')
    ax.fill_between(m.index, m['p25'], m['p75'],
                    alpha=0.30, color='C0', label='p25–p75 (middle half)')
    ax.plot(m.index, m['median'], color='C0', lw=2, marker='o', ms=4,
            label=f'{unit} median')
    if show_mean:
        ax.plot(m.index, m['mean'], color='C1', lw=1.2, ls='--',
                label=f'{unit} mean')

    if show_outliers and len(outliers):
        ax.scatter(outliers.index, outliers.values, s=14, marker='d',
                   facecolors='none', edgecolors='C0', lw=0.8, alpha=0.8,
                   zorder=4, label='Outlier days (beyond 1.5×IQR)')


def _format_season_axis(ax):
    """Tick every season start, labelled with the season name and month."""
    seasons = {12: 'Summer', 3: 'Autumn', 6: 'Winter', 9: 'Spring'}

    def season_label(x, pos):
        dt = mdates.num2date(x)
        return f"{seasons[dt.month]}\n{dt:%b %Y}"

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_major_formatter(FuncFormatter(season_label))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())


def soc_band_plot(df, col=None, date_col=None, period='month', show_mean=True,
                  show_outliers=True, show_sse=True, title=None, filename=None):
    """
    Median line with a shaded p10-p90 band, per month or per season.

    df            : DataFrame with daily values
    col           : name of the value column (defaults to the only/first numeric column)
    date_col      : name of the date column, if dates aren't already the index
    period        : 'month' or 'season' - how days are grouped
    show_mean     : also draw the period mean as a dashed line
    show_outliers : scatter the days falling outside that period's 1.5xIQR fences
    show_sse      : shade SSE events from data/sse_events.csv, coloured by category
    filename      : override the plot filename (defaults to one per period)
    """
    if period not in _PERIOD_SPECS:
        raise ValueError(f"period must be one of {list(_PERIOD_SPECS)}, got {period!r}")
    spec = _PERIOD_SPECS[period]
    unit = spec['label']

    s = _prepare_series(df, col, date_col)
    m, outliers = _period_summary(s, spec['freq'])

    # plot
    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_soc_band(ax, m, outliers, unit, show_mean, show_outliers)

    # lock the window to the data before shading events, so an SSE that runs
    # past either end of the series can't stretch the axis
    xlim = (m.index.min(), m.index.max() + spec['end'])
    ax.set_xlim(xlim)

    sse_handles = []
    if show_sse:
        sse_handles = _draw_sse_events(ax, load_sse_events(), xlim)

    _format_season_axis(ax)

    ax.set_ylabel("Fleet SOC (%)")
    ax.set_title(title or f'Fleet SOC trend (at ESROI start - 17:30): {unit.lower()} median with spread')
    ax.grid(alpha=0.3)

    # two legends: the SOC series on the plot, the event categories outside it
    main_legend = ax.legend(loc='best', fontsize=9)
    if sse_handles:
        ax.add_artist(main_legend)
        ax.legend(handles=sse_handles, title='SSE category', fontsize=9,
                  title_fontsize=9, loc='upper left',
                  bbox_to_anchor=(1.01, 1.0), frameon=False)

    fig.tight_layout()
    save_figure(fig, local_plots_dir / (filename or f"{spec['stem']}.png"))

    return fig, ax, m



def _merge_events(events, long=pd.Timedelta(days=1)):
    """Collapse overlapping same-day events of the same category into one.

    The register often carries several advisories for one situation - three
    separate heatwave entries on the same afternoon, say. Drawn as-is they
    overprint into a single thicker mark that reads as one event anyway, so
    merging them loses nothing and keeps the strip honest about how many
    distinct episodes there were.

    Events lasting `long` or more are never merged, and never absorb a short
    one. Without that rule a single wide event swallows everything inside it:
    the May-Oct 2024 outage alone spans 158 days and would hide the fourteen
    separate evening outages that fall within it.
    """
    duration = events['end'] - events['start']
    spans = events[duration >= long]
    short = events[duration < long]

    merged = []
    for cat, grp in short.groupby('category'):
        grp = grp.sort_values('start')
        start = end = None
        for _, e in grp.iterrows():
            if start is None:
                start, end = e['start'], e['end']
            elif e['start'] <= end:              # overlaps the run so far
                end = max(end, e['end'])
            else:
                merged.append((cat, start, end))
                start, end = e['start'], e['end']
        if start is not None:
            merged.append((cat, start, end))

    out = pd.concat([
        pd.DataFrame(merged, columns=['category', 'start', 'end']),
        spans[['category', 'start', 'end']],
    ], ignore_index=True)
    return out.sort_values('start').reset_index(drop=True)


def _draw_event_strip(ax, events, xlim, style='auto', min_span_frac=0.004):
    """One lane per SSE category, each event a bar (multi-day) or tick (single
    day) in its lane. Keeps events off the SOC axes entirely, so density in a
    busy week no longer paints over the data.

    style : 'auto'  - span if the event runs a day or more, tick if not
            'span'  - always a span, widened to stay visible if very short
            'line'  - always a tick at the start time
    """
    if style not in ('auto', 'span', 'line'):
        raise ValueError(f"style must be 'auto', 'span' or 'line', got {style!r}")

    lo, hi = xlim

    # the SOC series is tz-aware (Australia/Perth); the event times in the csv
    # are bare local times, so put them on the same clock before comparing
    if lo.tz is not None:
        events = events.copy()
        for c in ("start", "end"):
            events[c] = events[c].dt.tz_localize(lo.tz)

    events = events[(events['end'] >= lo) & (events['start'] <= hi)]

    # a lane per category present, in the palette's order so the strip reads
    # the same way across plots
    cats = [c for c in SSE_CATEGORY_COLORS if c in set(events['category'])]
    lane = {c: i for i, c in enumerate(cats)}

    # a single-day span is ~0.1% of a three-year axis, i.e. invisible - give
    # every span a floor wide enough to see
    lo_n, hi_n = mdates.date2num(lo), mdates.date2num(hi)
    min_w = (hi_n - lo_n) * min_span_frac

    for _, e in events.iterrows():
        y = lane[e['category']]
        color = SSE_CATEGORY_COLORS.get(e['category'], "#333333")
        start_n, end_n = mdates.date2num(e['start']), mdates.date2num(e['end'])

        if style == 'line' or (style == 'auto' and (e['end'] - e['start']) < pd.Timedelta(days=1)):
            # drawn taller than the bars and on top, so a short event inside a
            # long span stays visible
            ax.vlines(start_n, y - 0.36, y + 0.36, color=color, lw=1.6,
                      alpha=0.95, zorder=3)
        else:
            ax.barh(y, width=max(end_n - start_n, min_w), left=start_n,
                    height=0.5, color=color, alpha=0.5, lw=0, zorder=2)

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c.capitalize() for c in cats], fontsize=8)
    ax.set_ylim(len(cats) - 0.5, -0.5)   # first category at the top
    ax.tick_params(axis='y', length=0)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)

    return events


def soc_band_event_plot(df, col=None, date_col=None, period='month',
                        show_mean=True, show_outliers=True, event_style='auto',
                        merge_events=True, title=None, filename=None):
    """
    The same SOC band as soc_band_plot, with SSE events in their own strip
    below instead of shaded over the data.

    Same arguments as soc_band_plot, minus show_sse (the strip is the point
    here), plus:

    event_style  : 'auto' (span if a day or longer, tick if not), 'span', 'line'
    merge_events : collapse overlapping events of the same category into one

    Returns (fig, (soc_ax, event_ax), summary).
    """
    if period not in _PERIOD_SPECS:
        raise ValueError(f"period must be one of {list(_PERIOD_SPECS)}, got {period!r}")
    spec = _PERIOD_SPECS[period]
    unit = spec['label']

    s = _prepare_series(df, col, date_col)
    m, outliers = _period_summary(s, spec['freq'])

    fig, (ax, ax_ev) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True,
        gridspec_kw={'height_ratios': [4, 1], 'hspace': 0.08},
    )

    _draw_soc_band(ax, m, outliers, unit, show_mean, show_outliers)

    xlim = (m.index.min(), m.index.max() + spec['end'])
    ax.set_xlim(xlim)

    events = load_sse_events()
    if merge_events:
        events = _merge_events(events)
    _draw_event_strip(ax_ev, events, xlim, style=event_style)

    ax.set_ylabel("Fleet SOC (%)")
    ax.set_title(title or f'Fleet SOC trend (at ESROI start - 17:30): {unit.lower()} median with spread')
    ax.grid(alpha=0.3)
    ax.legend(loc='best', fontsize=9)

    ax_ev.set_ylabel("SSE", fontsize=9)
    ax_ev.grid(axis='x', alpha=0.3)
    _format_season_axis(ax_ev)

    save_figure(fig, local_plots_dir / (filename or f"{spec['stem']}_events.png"))

    return fig, (ax, ax_ev), m



# fig, ax, summary = soc_band_plot(fleet.at_time("17:30"))
# fig, ax, summary = soc_band_plot(fleet.at_time("17:30"), period="season", show_sse=False)
# fig, axes, summary = soc_band_event_plot(fleet.at_time("17:30"))
# fig, axes, summary = soc_band_event_plot(fleet.at_time("17:30"), period="season")
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

