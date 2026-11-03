import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pandas.tseries.frequencies import to_offset

from analysis.sse_events import load_sse_events
from tools.paths import local_plots_dir
from tools.plot_style import save_figure
from tools.seasons import SEASON_BY_MONTH


# one colour per SSE category, kept away from the blue/orange the band
# already uses so an event stripe never reads as part of the data
SSE_CATEGORY_COLORS = {
    "heatwave": "#D7191C",  # red
    "temperature": "#00A6A6",  # teal
    "outage": "#1A1A1A",  # near-black
    "multiple": "#7D3AC1",  # violet
    "other": "#4CAF50",  # green
}


def _draw_sse_events(ax, events, xlim):
    """Shade each SSE event on ax: a span if it runs over more than a day,
    a single line if not. Returns legend handles, one per category present."""
    lo, hi = xlim

    # the series may be tz-aware (Australia/Perth); the event times in the csv
    # are bare local times, so put them on the same clock before comparing
    if lo.tz is not None:
        events = events.copy()
        for c in ("start", "end"):
            events[c] = events[c].dt.tz_localize(lo.tz)

    drawn = set()
    for _, e in events.iterrows():
        if e["end"] < lo or e["start"] > hi:  # outside the plotted window
            continue
        color = SSE_CATEGORY_COLORS.get(e["category"], "#333333")
        if (e["end"] - e["start"]) >= pd.Timedelta(days=1):
            ax.axvspan(e["start"], e["end"], color=color, alpha=0.15, lw=0, zorder=0)
        else:
            ax.axvline(e["start"], color=color, alpha=0.40, lw=0.9, zorder=0)
        drawn.add(e["category"])

    order = [c for c in SSE_CATEGORY_COLORS if c in drawn]
    return [Patch(facecolor=SSE_CATEGORY_COLORS[c], alpha=0.5, label=c.capitalize()) for c in order]


# How the value column gets grouped before the band is drawn. "season" uses
# quarters starting in December, so the four buckets are exactly the southern
# seasons the x-axis already labels: Dec-Feb summer, Mar-May autumn,
# Jun-Aug winter, Sep-Nov spring.
_PERIOD_SPECS = {
    "month": {
        "freq": "MS",
        "label": "Monthly",
        "end": pd.offsets.MonthEnd(0),
    },
    "season": {
        "freq": "QS-DEC",
        "label": "Seasonal",
        "end": pd.offsets.QuarterEnd(startingMonth=2),
    },
}

# A period with less than this share of the values it should have - a partial
# first or last month, or a long data gap - is left out of the median and
# bands (a few days can't stand in for a month), and shaded grey instead.
MIN_COVERAGE = 0.5
LOW_COVERAGE_COLOR = "0.85"


def _period_coverage(s, freq):
    """Per period: the share of the values it should have that s actually has, and where it ends.

    "Should have" is the period's length over the series' usual spacing, so 5-min,
    hourly, daily and once-a-day series are all judged the same way."""
    counts = s.resample(freq).count()
    end = counts.index + to_offset(freq)
    spacing = s.index.to_series().diff().median()
    expected = (end - counts.index) / spacing
    return pd.DataFrame({"coverage": counts.to_numpy() / expected.to_numpy(), "end": end}, index=counts.index)


def _shade_low_coverage(ax, coverage, min_coverage=MIN_COVERAGE):
    """Grey span over each period left out for having too little data.

    coverage: frame with coverage and end columns, indexed by period start (from _period_coverage)
    Returns a legend handle for the shading, or None if no period was left out.
    """
    low = coverage[coverage["coverage"] < min_coverage]
    for start, row in low.iterrows():
        ax.axvspan(start, row["end"], color=LOW_COVERAGE_COLOR, lw=0, zorder=0)
    if low.empty:
        return None
    return Patch(facecolor=LOW_COVERAGE_COLOR, label=f"Under {min_coverage:.0%} of data, left out")


def _prepare_series(df, col=None, date_col=None):
    """One sorted, datetime-indexed Series out of whatever was passed in."""
    # accept a Series as well as a DataFrame
    if isinstance(df, pd.Series):
        df = df.to_frame(name=df.name if df.name is not None else "value")

    d = df.copy()

    # get a DatetimeIndex
    if date_col is not None:
        d[date_col] = pd.to_datetime(d[date_col])
        d = d.set_index(date_col)
    elif not isinstance(d.index, pd.DatetimeIndex):
        d.index = pd.to_datetime(d.index)
    d = d.sort_index()

    if col is None:
        col = d.select_dtypes("number").columns[0]
    return d[col]


def _period_summary(s, freq, min_coverage=MIN_COVERAGE):
    """Per-period median/mean/percentiles, plus the values that fall outside
    their own period's 1.5xIQR fences. Returns (summary frame, outlier values).

    s can be at any frequency (5-min, hourly, daily): every value in a period
    goes into that period's statistics. Periods with less than min_coverage of
    their values keep n, coverage and end but get NaN statistics and no outliers."""
    g = s.resample(freq)
    m = pd.DataFrame(
        {
            "median": g.median(),
            "mean": g.mean(),
            "p10": g.quantile(0.10),
            "p25": g.quantile(0.25),
            "p75": g.quantile(0.75),
            "p90": g.quantile(0.90),
            "min": g.min(),
            "max": g.max(),
            "n": g.count(),
        }
    )
    m = m.join(_period_coverage(s, freq))
    stats = ["median", "mean", "p10", "p25", "p75", "p90", "min", "max"]
    m.loc[m["coverage"] < min_coverage, stats] = float("nan")

    # Tukey fences off the period's own quartiles: a value is an outlier relative
    # to the period it sits in, so a quiet winter and a volatile summer each
    # get judged on their own spread.
    iqr = m["p75"] - m["p25"]
    m["lower_fence"] = m["p25"] - 1.5 * iqr
    m["upper_fence"] = m["p75"] + 1.5 * iqr

    # which row of m each value belongs to - m.index holds the period starts,
    # so the last start at or before a timestamp is that value's period.
    # A left-out period has NaN fences, and NaN comparisons are False, so it has no outliers.
    pos = m.index.searchsorted(s.index, side="right") - 1
    fences = m.iloc[pos]
    is_outlier = (s.values < fences["lower_fence"].values) | (s.values > fences["upper_fence"].values)
    outliers = s[is_outlier]
    m["n_outliers"] = outliers.resample(freq).count().reindex(m.index).fillna(0).astype(int)

    return m, outliers


def _draw_band(ax, m, outliers, unit, show_mean=True, show_outliers=True):
    """The series itself: p10-p90 and p25-p75 bands, median, mean, outliers."""
    ax.fill_between(m.index, m["p10"], m["p90"], alpha=0.18, color="C0", label="p10–p90")
    ax.fill_between(m.index, m["p25"], m["p75"], alpha=0.30, color="C0", label="p25–p75")
    ax.plot(m.index, m["median"], color="C0", lw=2, marker="o", ms=4, label=f"{unit} median")
    if show_mean:
        ax.plot(m.index, m["mean"], color="C1", lw=1.2, ls="--", label=f"{unit} mean")

    if show_outliers and len(outliers):
        ax.scatter(
            outliers.index,
            outliers.values,
            s=14,
            marker="d",
            facecolors="none",
            edgecolors="C0",
            lw=0.8,
            alpha=0.8,
            zorder=4,
            label="Outliers (beyond 1.5×IQR)",
        )


def _format_season_axis(ax):
    """Tick every season start, labelled with the season name and month."""

    def season_label(x, pos):
        dt = mdates.num2date(x)
        return f"{SEASON_BY_MONTH[dt.month]}\n{dt:%b %Y}"

    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[3, 6, 9, 12]))
    ax.xaxis.set_major_formatter(FuncFormatter(season_label))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())


def _title_and_legend_above(ax, title, handles=None, extra=None):
    """Set the title and put the legend in one row between the title and the plot, so it never covers data.

    handles: legend entries to use instead of the labelled artists on ax
    extra: one more entry to add at the end (e.g. from _shade_low_coverage), skipped if None
    """
    if handles is None:
        handles = ax.get_legend_handles_labels()[0]
    if extra is not None:
        handles = [*handles, extra]
    legend = ax.legend(
        handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=len(handles), fontsize=8.5, frameon=False
    )
    ax.set_title(title, pad=26)  # room for the legend row
    return legend


def trend_band_plot(
    df,
    col=None,
    date_col=None,
    period="month",
    show_mean=True,
    show_outliers=True,
    min_coverage=MIN_COVERAGE,
    ylabel=None,
    title=None,
    filename=None,
):
    """
    Median line with a shaded p10-p90 band, per month or per season.
    For SSE events alongside the band, use trend_band_event_plot.

    df: DataFrame or Series of values at any frequency (5-min, hourly, daily) -
        every value in a month/season goes into that period's median and bands
    col: name of the value column (defaults to the only/first numeric column)
    date_col: name of the date column, if dates aren't already the index
    period: 'month' or 'season' - how values are grouped
    show_mean: also draw the period mean as a dashed line
    show_outliers: scatter the values falling outside that period's 1.5xIQR fences
    min_coverage: periods with less than this share of their values are left out and shaded grey
    ylabel: y-axis label (defaults to the column name)
    title: plot title (defaults to "<column>: <period> median with spread")
    filename: plot filename (defaults to "<column>_<period>_trend.png")
    """
    if period not in _PERIOD_SPECS:
        raise ValueError(f"period must be one of {list(_PERIOD_SPECS)}, got {period!r}")
    spec = _PERIOD_SPECS[period]
    unit = spec["label"]

    s = _prepare_series(df, col, date_col)
    m, outliers = _period_summary(s, spec["freq"], min_coverage)

    # plot
    fig, ax = plt.subplots(figsize=(11, 5))
    _draw_band(ax, m, outliers, unit, show_mean, show_outliers)
    low_coverage_key = _shade_low_coverage(ax, m, min_coverage)

    ax.set_xlim(m.index.min(), m.index.max() + spec["end"])
    _format_season_axis(ax)

    ax.set_ylabel(ylabel or s.name)
    ax.grid(alpha=0.3)
    _title_and_legend_above(ax, title or f"{s.name}: {unit.lower()} median with spread", extra=low_coverage_key)

    fig.tight_layout()
    save_figure(fig, local_plots_dir / (filename or f"{s.name}_{period}_trend.png"))

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
    duration = events["end"] - events["start"]
    spans = events[duration >= long]
    short = events[duration < long]

    merged = []
    for cat, grp in short.groupby("category"):
        grp = grp.sort_values("start")
        start = end = None
        for _, e in grp.iterrows():
            if start is None:
                start, end = e["start"], e["end"]
            elif e["start"] <= end:  # overlaps the run so far
                end = max(end, e["end"])
            else:
                merged.append((cat, start, end))
                start, end = e["start"], e["end"]
        if start is not None:
            merged.append((cat, start, end))

    out = pd.concat(
        [
            pd.DataFrame(merged, columns=["category", "start", "end"]),
            spans[["category", "start", "end"]],
        ],
        ignore_index=True,
    )
    return out.sort_values("start").reset_index(drop=True)


def _draw_event_strip(ax, events, xlim, style="auto", min_span_frac=0.004):
    """One lane per SSE category, each event a bar (multi-day) or tick (single
    day) in its lane. Keeps events off the main axes entirely, so density in a
    busy week no longer paints over the data.

    style: 'auto' - span if the event runs a day or more, tick if not
        'span' - always a span, widened to stay visible if very short
        'line' - always a tick at the start time
    """
    if style not in ("auto", "span", "line"):
        raise ValueError(f"style must be 'auto', 'span' or 'line', got {style!r}")

    lo, hi = xlim

    # the series may be tz-aware (Australia/Perth); the event times in the csv
    # are bare local times, so put them on the same clock before comparing
    if lo.tz is not None:
        events = events.copy()
        for c in ("start", "end"):
            events[c] = events[c].dt.tz_localize(lo.tz)

    events = events[(events["end"] >= lo) & (events["start"] <= hi)]

    # a lane per category present, in the palette's order so the strip reads
    # the same way across plots
    cats = [c for c in SSE_CATEGORY_COLORS if c in set(events["category"])]
    lane = {c: i for i, c in enumerate(cats)}

    # a single-day span is ~0.1% of a three-year axis, i.e. invisible - give
    # every span a floor wide enough to see
    lo_n, hi_n = mdates.date2num(lo), mdates.date2num(hi)
    min_w = (hi_n - lo_n) * min_span_frac

    for _, e in events.iterrows():
        y = lane[e["category"]]
        color = SSE_CATEGORY_COLORS.get(e["category"], "#333333")
        start_n, end_n = mdates.date2num(e["start"]), mdates.date2num(e["end"])

        if style == "line" or (style == "auto" and (e["end"] - e["start"]) < pd.Timedelta(days=1)):
            # drawn taller than the bars and on top, so a short event inside a
            # long span stays visible
            ax.vlines(start_n, y - 0.36, y + 0.36, color=color, lw=1.6, alpha=0.95, zorder=3)
        else:
            ax.barh(
                y, width=max(end_n - start_n, min_w), left=start_n, height=0.5, color=color, alpha=0.5, lw=0, zorder=2
            )

    ax.set_yticks(range(len(cats)))
    ax.set_yticklabels([c.capitalize() for c in cats], fontsize=8)
    ax.set_ylim(len(cats) - 0.5, -0.5)  # first category at the top
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)

    return events


def trend_band_event_plot(
    df,
    col=None,
    date_col=None,
    period="month",
    show_mean=True,
    show_outliers=True,
    event_style="auto",
    merge_events=True,
    min_coverage=MIN_COVERAGE,
    ylabel=None,
    title=None,
    filename=None,
):
    """
    The same band as trend_band_plot, with SSE events in their own strip
    below instead of shaded over the data.

    Same arguments as trend_band_plot, minus show_sse (the strip is the point
    here), plus:

    event_style: 'auto' (span if a day or longer, tick if not), 'span', 'line'
    merge_events: collapse overlapping events of the same category into one

    Returns (fig, (main_ax, event_ax), summary).
    """
    if period not in _PERIOD_SPECS:
        raise ValueError(f"period must be one of {list(_PERIOD_SPECS)}, got {period!r}")
    spec = _PERIOD_SPECS[period]
    unit = spec["label"]

    s = _prepare_series(df, col, date_col)
    m, outliers = _period_summary(s, spec["freq"], min_coverage)

    fig, (ax, ax_ev) = plt.subplots(
        2,
        1,
        figsize=(11, 6.2),
        sharex=True,
        gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08},
    )

    _draw_band(ax, m, outliers, unit, show_mean, show_outliers)
    low_coverage_key = _shade_low_coverage(ax, m, min_coverage)

    xlim = (m.index.min(), m.index.max() + spec["end"])
    ax.set_xlim(xlim)

    events = load_sse_events()
    if merge_events:
        events = _merge_events(events)
    _draw_event_strip(ax_ev, events, xlim, style=event_style)

    ax.set_ylabel(ylabel or s.name)
    ax.grid(alpha=0.3)
    _title_and_legend_above(ax, title or f"{s.name}: {unit.lower()} median with spread", extra=low_coverage_key)

    ax_ev.set_ylabel("SSE", fontsize=9)
    ax_ev.grid(axis="x", alpha=0.3)
    _format_season_axis(ax_ev)

    save_figure(fig, local_plots_dir / (filename or f"{s.name}_{period}_trend_events.png"))

    return fig, (ax, ax_ev), m


def stacked_median_plot(panels, period="month", min_coverage=MIN_COVERAGE, title=None, filename=None):
    """
    Period median of several daily series, one panel each, stacked on a shared
    time axis. Each panel keeps its own y-scale, so series of very different
    size (system demand vs battery discharge) can be read month against month
    without one flattening the other.

    panels: list of (series, ylabel, color), drawn top to bottom
    period: 'month' or 'season' - how days are grouped
    min_coverage: periods with less than this share of their values are left out and shaded grey
    title: figure title
    filename: plot filename (defaults to "stacked_<period>_median.png")

    Returns (fig, axes, medians) - medians has one column per panel, named by its ylabel.
    """
    if period not in _PERIOD_SPECS:
        raise ValueError(f"period must be one of {list(_PERIOD_SPECS)}, got {period!r}")
    spec = _PERIOD_SPECS[period]

    fig, axes = plt.subplots(
        len(panels),
        1,
        figsize=(11, 1 + 2.6 * len(panels)),
        sharex=True,
        squeeze=False,
        gridspec_kw={"hspace": 0.1},
    )
    axes = axes[:, 0]

    medians = {}
    low_coverage_key = None
    for ax, (s, ylabel, color) in zip(axes, panels):
        s = _prepare_series(s)
        coverage = _period_coverage(s, spec["freq"])
        med = s.resample(spec["freq"]).median().where(coverage["coverage"] >= min_coverage)
        ax.plot(med.index, med, color=color, lw=2, marker="o", ms=4)
        low_coverage_key = _shade_low_coverage(ax, coverage, min_coverage) or low_coverage_key
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.3)
        medians[ylabel] = med
    medians = pd.DataFrame(medians)

    # sharex, so setting the window on one panel sets it on all
    axes[0].set_xlim(medians.index.min(), medians.index.max() + spec["end"])
    _format_season_axis(axes[-1])

    if low_coverage_key is not None:
        axes[0].legend(handles=[low_coverage_key], loc="upper left", fontsize=8.5, frameon=False)
    if title:
        axes[0].set_title(title)

    save_figure(fig, local_plots_dir / (filename or f"stacked_{period}_median.png"))

    return fig, axes, medians


def mark_period_medians(ax, s, periods, value_format="{:,.0f}", color="0.25", legend_loc="upper left"):
    """Dashed line at the median of s over each period, spanning that period, with the
    medians and the change from the first period to the last as a legend entry.

    s: the series the panel was drawn from (any frequency)
    periods: [(start, end), ...] - compare like with like, e.g. the same months a year apart,
        so seasonality doesn't read as change
    value_format: how a median is written, e.g. "{:,.0f} MWh" or "{:.1f}%"
    legend_loc: where the legend goes - the default sits in the headroom added above the data;
        move it if something else (e.g. mark_commissioning labels) is already up there

    Returns the medians, one per period.
    """
    tz = s.index.tz
    medians = []
    for start, end in periods:
        a, b = pd.Timestamp(start, tz=tz), pd.Timestamp(end, tz=tz)
        median = s.loc[a:b].median()
        medians.append(median)
        ax.hlines(median, a, b, colors=color, linestyles="--", lw=1.4, zorder=3)

    parts = [f"{pd.Timestamp(a):%b %Y}–{pd.Timestamp(b):%b %Y}: {value_format.format(m)}"
             for (a, b), m in zip(periods, medians, strict=True)]
    change = (medians[-1] / medians[0] - 1) * 100
    key = Line2D([], [], color=color, ls="--", lw=1.4, label=f"Median {' → '.join(parts)} ({change:+.0f}%)")

    # room above the data for the legend, so it can't sit on a point
    bottom, top = ax.get_ylim()
    ax.set_ylim(bottom, top + 0.2 * (top - bottom))
    # plain white backing (no border) so lines underneath, e.g. commissioning markers, don't run through the text
    ax.legend(handles=[key], loc=legend_loc, fontsize=8.5, facecolor="white", edgecolor="none", framealpha=1)

    return medians


def mark_commissioning(ax, dates, capacity):
    """Dashed line at each unit's commissioning date, labelled with the MWh it adds.

    dates: {code: commissioning timestamp}, from constants.commissioning_dates(...)
    capacity: {code: MWh}, e.g. battery_capacity_MWh

    Units commissioned at or before the left edge of the axis aren't marked:
    for those the "commissioning date" is just where the data starts.
    """
    left = ax.get_xlim()[0]
    for code, start in sorted(dates.items(), key=lambda kv: kv[1]):
        if code not in capacity:  # non-batteries have no MWh rating
            continue
        x = mdates.date2num(start)
        if x <= left:
            continue
        ax.axvline(x, color="0.5", ls="--", lw=0.9, zorder=0)
        # nudged 2pt left so the text doesn't sit on its own line
        ax.annotate(
            f"{code} +{capacity[code]:,} MWh",
            xy=(x, 0.97),
            xycoords=ax.get_xaxis_transform(),
            xytext=(-2, 0),
            textcoords="offset points",
            rotation=90,
            va="top",
            ha="right",
            fontsize=7.5,
            color="0.35",
        )


def period_box_plot(
    df,
    col=None,
    date_col=None,
    period="month",
    whis=(10, 90),
    ylim=None,
    min_coverage=MIN_COVERAGE,
    ylabel=None,
    title=None,
    filename=None,
):
    """
    One box per month (or season) of daily values, on the same time axis as
    the trend plots.

    df: DataFrame or Series with daily values
    col: name of the value column (defaults to the only/first numeric column)
    date_col: name of the date column, if dates aren't already the index
    period: 'month' or 'season' - how days are grouped
    whis: whisker percentiles (default 10th-90th, same as the trend plots' outer band);
        days beyond them are drawn as dots
    ylim: (bottom, top) of the y-axis, e.g. (0, 100) for a % (defaults to fitting the data)
    min_coverage: periods with less than this share of their values get no box and are shaded grey
    ylabel: y-axis label (defaults to the column name)
    title: plot title
    filename: plot filename (defaults to "<column>_<period>_box.png")

    Returns (fig, ax, summary) - summary is the per-period median/percentiles/n.
    """
    if period not in _PERIOD_SPECS:
        raise ValueError(f"period must be one of {list(_PERIOD_SPECS)}, got {period!r}")
    spec = _PERIOD_SPECS[period]

    s = _prepare_series(df, col, date_col)
    m, _ = _period_summary(s, spec["freq"], min_coverage)

    enough = m.index[m["coverage"] >= min_coverage]
    groups = [(start, g.dropna()) for start, g in s.resample(spec["freq"]) if start in enough]
    groups = [(start, g) for start, g in groups if len(g)]

    # each box sits in the middle of its period and covers 70% of it, so the
    # season ticks (at period starts) fall in the gaps between boxes
    starts = [mdates.date2num(start) for start, _ in groups]
    ends = [mdates.date2num(start + spec["end"]) + 1 for start, _ in groups]
    positions = [(a + b) / 2 for a, b in zip(starts, ends)]
    widths = [0.7 * (b - a) for a, b in zip(starts, ends)]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.boxplot(
        [g.to_numpy() for _, g in groups],
        positions=positions,
        widths=widths,
        whis=whis,
        manage_ticks=False,  # keep the date axis, not one tick per box
        patch_artist=True,
        boxprops={"facecolor": "C0", "alpha": 0.5},
        # median thicker than the whiskers, so the two read apart in the legend too
        medianprops={"color": "black", "lw": 2},
        whiskerprops={"color": "0.4", "lw": 1},
        capprops={"color": "0.4", "lw": 1},
        flierprops={"marker": "o", "markersize": 3, "markerfacecolor": "C0", "markeredgecolor": "none", "alpha": 0.75},
    )

    # the window spans every period with data, left-out ones included, so their shading shows
    ax.set_xlim(mdates.date2num(m.index.min()), mdates.date2num(m["end"].max()))
    low_coverage_key = _shade_low_coverage(ax, m, min_coverage)
    if ylim is not None:
        ax.set_ylim(ylim)
    _format_season_axis(ax)

    ax.set_ylabel(ylabel or s.name)
    ax.grid(alpha=0.3)

    # boxplot artists carry no labels, so the legend gets stand-ins that look like each part
    lo, hi = whis
    handles = [
        Patch(facecolor="C0", alpha=0.5, edgecolor="black", label="p25–p75 (middle half)"),
        Line2D([], [], color="black", lw=2, label=f"{spec['label']} median"),
        Line2D([], [], color="0.4", lw=1, label=f"p{lo}–p{hi} (whiskers)"),
        Line2D([], [], ls="none", marker="o", markersize=3, color="C0", alpha=0.75, label=f"Days beyond p{lo}–p{hi}"),
    ]
    _title_and_legend_above(ax, title or f"{s.name}: {spec['label'].lower()} distribution", handles, low_coverage_key)

    fig.tight_layout()
    save_figure(fig, local_plots_dir / (filename or f"{s.name}_{period}_box.png"))

    return fig, ax, m
