"""Rewritten diagnostic/analysis plots: a 4-panel single-axis-per-metric day
view, a headroom (stored vs required energy) view that carries the adequacy
argument, and a whole-record entry-energy adequacy scatter.

Reuses the same raw-data loaders and cleaning already in this repo (see
main.py, tools/df_management.py) rather than inventing a new schema. Keeps
plot_heatmap.py/plot_distributions.py/plot_soc.py untouched - this is an
addition, not a replacement.
"""

from collections.abc import Callable

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib import patheffects
from matplotlib.colors import SymLogNorm, TwoSlopeNorm

from main import _extract_case_input_data, _extract_power_and_price
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, battery_capacity_MW, battery_codes
from tools.df_management import clean_charge_level_df, derive_capacity_from_observed_max
from tools.paths import repo_plots_dir, repo_processed_data_dir
from tools.plot_style import UNIT_COLORS, save_figure

_CLEARED_COLOR = "#2a9d5c"
_SHORT_COLOR = "#d1495b"
_SHORTFALL_FILL = "#d1495b"

_data_cache: dict[str, pd.DataFrame | pd.Series] = {}
_capacity_cache: dict[str, float] | None = None


def _gap_missing_telemetry(
    s: pd.Series, zero_minutes: int = 30, frozen_minutes: int = 60, freq_minutes: int = 5
) -> pd.Series:
    """A SCADA feed stuck at exactly 0, or stuck at any other constant
    value for a long stretch, is almost certainly missing/frozen telemetry
    rather than a real reading (an empty battery doesn't usually sit dead
    flat at 0.000 for hours). Runs of exact zero longer than `zero_minutes`,
    or runs of ANY constant value longer than `frozen_minutes`, get gapped
    to NaN so the line breaks instead of drawing a fake flat segment."""
    s = s.copy()
    zero_run_len = max(1, round(zero_minutes / freq_minutes))
    frozen_run_len = max(1, round(frozen_minutes / freq_minutes))

    same_as_prev = s.eq(s.shift())
    run_id = (~same_as_prev).cumsum()
    run_sizes = s.groupby(run_id).transform("size")

    is_long_zero_run = (s == 0) & (run_sizes >= zero_run_len)
    is_long_frozen_run = run_sizes >= frozen_run_len
    s[is_long_zero_run | is_long_frozen_run] = float("nan")
    return s


def _load_data(zero_minutes: int = 30, frozen_minutes: int = 60):
    """Load (from the data/processed_data parquet cache if present, else
    extract fresh - see main.py) and lightly clean the raw SOC/demand/power/
    price data. Cached in-process so a batch of plot calls only loads once."""
    cache_key = (zero_minutes, frozen_minutes)
    if _data_cache.get("_key") == cache_key:
        return _data_cache["soc"], _data_cache["power"], _data_cache["demand"], _data_cache["price"]

    case_input_data = _extract_case_input_data()
    raw_soc_df, demand_df = case_input_data["charge_level"], case_input_data["demand"]
    power_df, price_df = _extract_power_and_price()

    soc_df = clean_charge_level_df(raw_soc_df)
    for code in battery_codes:
        if code in soc_df:
            soc_df[code] = _gap_missing_telemetry(soc_df[code], zero_minutes, frozen_minutes)

    demand = demand_df["dispatchCondition.demand"]
    price = price_df["energy_price"]

    _data_cache.update(_key=cache_key, soc=soc_df, power=power_df, demand=demand, price=price)
    return soc_df, power_df, demand, price


def _get_soc_capacity() -> dict[str, float]:
    """Same empirically-derived-from-observed-max capacity main.py uses for
    the SOC % columns (see tools/df_management.py) - not the documented
    battery_capacity_MWh, which has been wrong before (KWINANA_ESR2).
    Cached in-process since it requires the full record's SOC data."""
    global _capacity_cache
    if _capacity_cache is None:
        soc_df, _, _, _ = _load_data()
        _capacity_cache = derive_capacity_from_observed_max(soc_df)
    return _capacity_cache


def _style_axes(ax: plt.Axes):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25, linewidth=0.6)


def _shade_peak_esroi(ax: plt.Axes, day: str, tz):
    start = pd.Timestamp(f"{day} {PEAK_ESROI_START}", tz=tz)
    end = pd.Timestamp(f"{day} {PEAK_ESROI_END}", tz=tz)
    ax.axvspan(start, end, color="gold", alpha=0.15, zorder=0)


def _slice_day(data: pd.DataFrame | pd.Series, day: str):
    try:
        sliced = data.loc[day]
    except KeyError:
        return None
    return None if sliced.empty else sliced


def _draw_day_panels(axes, day: str, soc_unit: str, units: list[str]):
    """Draw stored energy / power / demand / price for one day onto the
    given (ax_soc, ax_power, ax_demand, ax_price) axes - the shared core of
    plot_day and plot_day_comparison, so a single-day figure and a
    side-by-side comparison figure never draw a panel two different ways.
    Returns the day's tz, or None (and draws a "no data" marker) if there's
    no SOC data for this day."""
    ax_soc, ax_power, ax_demand, ax_price = axes
    soc_df, power_df, demand, price = _load_data()

    day_soc = _slice_day(soc_df, day)
    day_power = _slice_day(power_df, day)
    day_demand = _slice_day(demand, day)
    day_price = _slice_day(price, day)

    if day_soc is None:
        print(f"{day}: no SOC data")
        ax_soc.text(0.5, 0.5, f"no data\n{day}", ha="center", va="center", transform=ax_soc.transAxes, color="grey")
        for ax in axes:
            _style_axes(ax)
        return None

    tz = day_soc.index.tz

    if soc_unit == "pct":
        capacity = _get_soc_capacity()
        for code in units:
            if code in day_soc:
                ax_soc.plot(
                    day_soc.index, day_soc[code] / capacity[code] * 100, color=UNIT_COLORS[code], linewidth=1, label=code
                )
        ax_soc.set_ylim(0, 100)
        ax_soc.set_ylabel("State of charge\n(%)")
    else:
        for code in units:
            if code in day_soc:
                ax_soc.plot(day_soc.index, day_soc[code], color=UNIT_COLORS[code], linewidth=1, label=code)
        ax_soc.set_ylim(bottom=0)
        ax_soc.set_ylabel("Stored energy\n(MWh)")

    if day_power is not None:
        for code in units:
            if code in day_power:
                ax_power.plot(day_power.index, day_power[code], color=UNIT_COLORS[code], linewidth=1, label=code)
    ax_power.axhline(0, color="grey", linewidth=0.8, zorder=0)
    ax_power.set_ylabel("Power (MW)\n+ discharge / - charge")

    if day_demand is not None:
        ax_demand.plot(day_demand.index, day_demand.to_numpy(), color="black", linewidth=1)
    ax_demand.set_ylabel("Demand (MW)")

    if day_price is not None:
        ax_price.plot(day_price.index, day_price.to_numpy(), color="black", linewidth=1)
    ax_price.set_ylabel("Price ($/MWh)")
    ax_price.set_xlabel("Time (AWST)")

    for ax in axes:
        _style_axes(ax)
        _shade_peak_esroi(ax, day, tz)

    # pin x-limits to exactly this day - matplotlib's default 5% margin
    # otherwise bleeds into the unplotted previous/next day and mislabels
    # the first/last tick as an hour with no actual data behind it
    ax_price.set_xlim(day_soc.index.min(), day_soc.index.max())
    ax_price.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    for ax in (ax_soc, ax_power, ax_demand):
        plt.setp(ax.get_xticklabels(), visible=False)

    return tz


def plot_day(day: str, units: list[str] | None = None, soc_unit: str = "mwh"):
    """Four stacked single-axis panels for one day: stored energy (MWh or %
    of rated capacity, per `soc_unit`), power (MW), demand (MW), price
    ($/MWh). No twinx anywhere - different scales get different panels,
    never a second y-axis sharing one frame.

    `soc_unit="mwh"` (default) plots absolute stored energy - saved as
    <day>_day.png. `soc_unit="pct"` plots % of each unit's empirically
    derived rated capacity (see _get_soc_capacity) - saved as
    <day>_day_pct.png. Generate both if you want them side by side; nothing
    else on the figure changes between the two."""
    if soc_unit not in ("mwh", "pct"):
        raise ValueError(f"soc_unit must be 'mwh' or 'pct', got {soc_unit!r}")

    units = units or battery_codes
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(12, 11.5), gridspec_kw={"hspace": 0.15, "top": 0.9})
    ax_soc = axes[0]

    tz = _draw_day_panels(axes, day, soc_unit, units)
    if tz is None:
        plt.close(fig)
        return None

    # panels 1 and 2 share the same unit -> colour mapping, so one legend
    # in the header area (above panel 1, never over data) covers both
    # instead of two duplicate legends each risking overlap with a line
    if len(units) >= 2:
        handles, labels = ax_soc.get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncols=6, fontsize="small", frameon=False
        )

    fig.suptitle(f"{day}", y=1.0)
    fig.align_ylabels(list(axes))
    suffix = "_pct" if soc_unit == "pct" else ""
    save_figure(fig, repo_plots_dir / f"{day}_day{suffix}.png")
    return fig


def plot_day_comparison(day: str, soc_unit: str = "mwh", days_before: int = 7, units: list[str] | None = None):
    """Two of plot_day's 4-panel stacks side by side in one figure: the day
    `days_before` days earlier on the left, `day` itself on the right (left
    to right is chronological order). Each row (SOC, power, demand, price)
    shares one y-axis across both columns, so the comparison reflects
    actual magnitude differences rather than each panel's own auto-scaling.
    Saved as <day>_vs_week_before[_pct].png."""
    if soc_unit not in ("mwh", "pct"):
        raise ValueError(f"soc_unit must be 'mwh' or 'pct', got {soc_unit!r}")

    units = units or battery_codes
    compare_day = (pd.Timestamp(day) - pd.Timedelta(days=days_before)).strftime("%Y-%m-%d")

    fig, axes = plt.subplots(
        4, 2, sharex="col", sharey="row", figsize=(20, 11.5), gridspec_kw={"hspace": 0.15, "wspace": 0.08, "top": 0.93}
    )
    left_axes = list(axes[:, 0])
    right_axes = list(axes[:, 1])

    tz_left = _draw_day_panels(left_axes, compare_day, soc_unit, units)
    tz_right = _draw_day_panels(right_axes, day, soc_unit, units)

    if tz_left is None and tz_right is None:
        print(f"{day}: no data for either {compare_day} or {day}, skipping plot_day_comparison")
        plt.close(fig)
        return None

    # sharey='row' already ties the scales together - repeating the label
    # and tick numbers on the right column would just be clutter
    for ax in right_axes:
        ax.set_ylabel("")
        plt.setp(ax.get_yticklabels(), visible=False)

    axes[0, 0].set_title(f"{compare_day}  (week before)", fontsize="medium")
    axes[0, 1].set_title(f"{day}  (stress event)", fontsize="medium")

    if len(units) >= 2:
        handles, labels = ([], [])
        for ax in (axes[0, 0], axes[0, 1]):
            h, l = ax.get_legend_handles_labels()
            if h:
                handles, labels = h, l
                break
        if handles:
            fig.legend(
                handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncols=6, fontsize="small", frameon=False
            )

    fig.suptitle(f"{day} vs {days_before} days before", y=1.045)
    fig.align_ylabels(left_axes)

    suffix = "_pct" if soc_unit == "pct" else ""
    save_figure(fig, repo_plots_dir / "week-before-comparison" / f"{day}_vs_week_before{suffix}.png")
    return fig


def _default_obligation_mw() -> float:
    return sum(battery_capacity_MW.values())


def plot_headroom(day: str, obligation_mw: float | Callable[[str], float] | None = None):
    """One panel, MWh: fleet stored energy (solid) vs the energy the fleet
    needs right now to hold `obligation_mw` through to Peak ESROI's close
    (dashed, sloping to zero at close, undefined outside the window).
    `obligation_mw` is a placeholder pending the real ESROD definition -
    pass a fixed number or a callable(day) -> MW to swap it without
    touching this figure's code. Returns (fig, shortfall_series, onset_time)
    so the same run can be aggregated across every day in the record -
    shortfall_series is NaN outside the window and wherever stored >=
    required; onset_time is the first clock time within the window where
    stored < required, or None if the day never fell short."""
    soc_df, _, _, _ = _load_data()
    day_soc = _slice_day(soc_df, day)
    if day_soc is None:
        print(f"{day}: no SOC data, skipping plot_headroom")
        return None, pd.Series(dtype=float), None

    tz = day_soc.index.tz
    if callable(obligation_mw):
        total_mw = obligation_mw(day)
    elif obligation_mw is not None:
        total_mw = obligation_mw
    else:
        total_mw = _default_obligation_mw()

    fleet_stored = day_soc[[c for c in battery_codes if c in day_soc]].sum(axis=1, min_count=1)

    window_start = pd.Timestamp(f"{day} {PEAK_ESROI_START}", tz=tz)
    window_end = pd.Timestamp(f"{day} {PEAK_ESROI_END}", tz=tz)
    in_window = (fleet_stored.index >= window_start) & (fleet_stored.index <= window_end)

    hours_remaining = (window_end - fleet_stored.index).total_seconds() / 3600
    required = pd.Series(hours_remaining * total_mw, index=fleet_stored.index)
    required[~in_window] = float("nan")

    shortfall = (required - fleet_stored).where(in_window)
    is_short = shortfall > 0
    onset_time = shortfall.index[is_short].min() if is_short.any() else None

    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(fleet_stored.index, fleet_stored, color="black", linewidth=1.3, label="fleet stored energy")
    ax.plot(required.index, required, color="black", linewidth=1.3, linestyle="--", label="required to hold to close")
    ax.fill_between(
        fleet_stored.index,
        fleet_stored,
        required,
        where=is_short.fillna(False),
        color=_SHORTFALL_FILL,
        alpha=0.25,
        label="shortfall",
        interpolate=True,
    )

    if onset_time is not None:
        ax.axvline(onset_time, color=_SHORTFALL_FILL, linestyle=":", linewidth=1.2)
        # anchor near the actual onset point on the required-energy line,
        # not the top of the axes - the legend also lives up there, and an
        # onset near window-open (a common case, since the obligation MW is
        # still a placeholder - see docstring) would collide with it
        ax.annotate(
            f"short from {onset_time.strftime('%H:%M')}",
            xy=(onset_time, required.loc[onset_time]),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize="small",
            color=_SHORTFALL_FILL,
        )

    ax.set_ylim(bottom=0)
    ax.set_xlim(fleet_stored.index.min(), fleet_stored.index.max())
    ax.set_ylabel("Energy (MWh)")
    ax.set_xlabel("Time (AWST)")
    ax.set_title(f"Fleet headroom vs obligation ({total_mw:.0f} MW to {PEAK_ESROI_END}) — {day}")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
    ax.legend(loc="upper left", fontsize="small")
    _style_axes(ax)

    save_figure(fig, repo_plots_dir / f"{day}_headroom.png")
    return fig, shortfall, onset_time


def plot_entry_energy():
    """One dot per day across the full record: fleet stored energy at Peak
    ESROI's start (17:30), coloured by whether it cleared the obligation
    that day. The requirement is a dashed step line, not a flat line - it's
    computed per day from only the units that actually have telemetry that
    day, so it steps up as each unit commissions instead of assuming the
    full fleet existed from day one."""
    soc_df, _, _, _ = _load_data()

    at_start = soc_df.at_time(PEAK_ESROI_START)
    codes_present = [c for c in battery_codes if c in at_start]
    fleet_stored = at_start[codes_present].sum(axis=1, min_count=1)

    window_hours = (
        pd.Timestamp(f"2000-01-01 {PEAK_ESROI_END}") - pd.Timestamp(f"2000-01-01 {PEAK_ESROI_START}")
    ).total_seconds() / 3600

    daily_active = soc_df[codes_present].notna().resample("D").max()
    required_mw_by_day = daily_active.apply(
        lambda row: sum(battery_capacity_MW[c] for c in codes_present if row[c]), axis=1
    )
    required_energy = required_mw_by_day.reindex(fleet_stored.index.normalize()) * window_hours
    required_energy.index = fleet_stored.index

    cleared = fleet_stored >= required_energy
    pct_short = 100 * (~cleared).sum() / len(cleared) if len(cleared) else 0.0

    fig, ax = plt.subplots(figsize=(13, 5.5))
    colors = cleared.map({True: _CLEARED_COLOR, False: _SHORT_COLOR})
    ax.scatter(fleet_stored.index, fleet_stored, c=colors, s=6, linewidths=0, zorder=2)
    ax.step(required_energy.index, required_energy, where="post", color="black", linestyle="--", linewidth=1, label="requirement", zorder=1)

    handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color=_CLEARED_COLOR, label="cleared obligation"),
        plt.Line2D([0], [0], marker="o", linestyle="", color=_SHORT_COLOR, label="short of obligation"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="requirement (steps as units commission)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize="small")

    ax.set_ylabel(f"Stored energy at {PEAK_ESROI_START} (MWh)")
    ax.set_xlabel("Date")
    ax.set_title(f"Fleet entry energy vs obligation — {pct_short:.1f}% of days short")
    _style_axes(ax)

    save_figure(fig, repo_plots_dir / "entry_energy.png")
    return fig


def _load_bidstack() -> pd.DataFrame | None:
    """Load bidstack.parquet if it's been produced yet (see main.py's
    _extract_case_input_data). Deliberately NOT wired into _load_data()'s
    cache - calling that would trigger main.py's full extraction path for
    any missing field, which is the wrong thing to do if that extraction
    is already running elsewhere (double-walks the corpus, races on the
    output file). Returns None with a clear message if it doesn't exist yet."""
    path = repo_processed_data_dir / "bidstack.parquet"
    if not path.exists():
        print(f"{path} doesn't exist yet - run main.py to produce it")
        return None
    return pd.read_parquet(path)


def _draw_bidstack_panel(ax: plt.Axes, day_code_bidstack: pd.DataFrame, norm, cmap: str = "turbo"):
    """One battery's bid stack across one day: x = dispatch interval, bars
    stacked by tranche (cumulative MW, lowest-tranche-number at the
    bottom - tranches are submitted in ascending price order, so this
    runs cheapest/most-negative-price at the bottom to priciest at the
    top), coloured by that tranche's submitted price. Many tranches have
    zero quantity (pure price breakpoints with no incremental capacity
    attached) and simply don't draw a visible segment. Returns the day's
    tz, or None if there's no data for this battery/day."""
    if day_code_bidstack.empty:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes, color="grey", fontsize="small")
        return None

    df = day_code_bidstack.sort_values(["dispatch_interval", "tranche"]).copy()
    df["bottom"] = df.groupby("dispatch_interval")["quantity"].cumsum() - df["quantity"]

    colormap = plt.get_cmap(cmap)
    colors = colormap(norm(df["submitted_price"].to_numpy()))

    width = 5 / (24 * 60)  # 5-minute dispatch interval, in matplotlib date units (days)
    # thin white edge on every segment - without it, adjacent tranches (and
    # adjacent 5-min columns) with similar colours just blend into one
    # another with no visible boundary
    ax.bar(
        df["dispatch_interval"],
        df["quantity"],
        bottom=df["bottom"],
        width=width,
        color=colors,
        align="edge",
        edgecolor="white",
        linewidth=0.4,
    )
    ax.axhline(0, color="grey", linewidth=0.8, zorder=2)

    return df["dispatch_interval"].dt.tz


_SOC_LINE_COLOR = "black"
_CLEARING_PRICE_LINE_COLOR = "#FFD400"


def _overlay_soc_and_price(ax: plt.Axes, soc: pd.Series | None, price: pd.Series | None, show_ticks: bool):
    """Twin-axis overlay of stored energy (SOC, MWh) and market clearing
    price ($/MWh) on top of a bid-stack panel. Deliberately breaks plot_day's
    "no twinx" rule - here the overlay IS the point (see how SOC and where
    the market cleared line up against this battery's own offer stack), not
    an accident of cramming unrelated series onto one frame. Both lines get
    a high-contrast stroke outline so they stay legible against the
    colour-coded bars underneath regardless of bar colour. `show_ticks`
    only draws axis numbers/labels (kept off half the grid - see caller - to
    cut clutter; the lines themselves always draw)."""
    ax_soc = ax.twinx()
    if soc is not None and not soc.empty:
        ax_soc.plot(
            soc.index,
            soc.to_numpy(),
            color=_SOC_LINE_COLOR,
            linewidth=2,
            zorder=6,
            path_effects=[patheffects.Stroke(linewidth=3.6, foreground="white"), patheffects.Normal()],
        )
    ax_soc.set_ylim(bottom=0)
    ax_soc.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    if show_ticks:
        ax_soc.set_ylabel("SOC (MWh)", fontsize="x-small", color=_SOC_LINE_COLOR)
        ax_soc.tick_params(axis="y", labelsize="x-small", colors=_SOC_LINE_COLOR)
    else:
        ax_soc.set_yticklabels([])
        ax_soc.tick_params(axis="y", length=0)

    ax_price = ax.twinx()
    ax_price.spines["right"].set_position(("axes", 1.14))
    if price is not None and not price.empty:
        ax_price.plot(
            price.index,
            price.to_numpy(),
            color=_CLEARING_PRICE_LINE_COLOR,
            linewidth=1.8,
            linestyle="--",
            zorder=6,
            path_effects=[patheffects.Stroke(linewidth=3.2, foreground="black"), patheffects.Normal()],
        )
    ax_price.yaxis.set_major_locator(mticker.MaxNLocator(nbins=3))
    if show_ticks:
        ax_price.set_ylabel("Clearing price ($/MWh)", fontsize="x-small", color="#8a6800")
        ax_price.tick_params(axis="y", labelsize="x-small", colors="#8a6800")
        ax_price.spines["right"].set_visible(True)
    else:
        ax_price.set_yticklabels([])
        ax_price.tick_params(axis="y", length=0)
        ax_price.spines["right"].set_visible(False)


def _build_price_norm(color_mode: str, combined: pd.DataFrame, price_clip: float, symlog_linthresh: float):
    abs_max = combined["submitted_price"].abs().max()
    if not abs_max or pd.isna(abs_max):
        abs_max = 1.0

    if color_mode == "clip":
        return TwoSlopeNorm(vmin=-price_clip, vcenter=0, vmax=price_clip), "both"
    if color_mode == "minmax":
        return TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max), "neither"
    if color_mode == "symlog":
        return SymLogNorm(linthresh=symlog_linthresh, vmin=-abs_max, vmax=abs_max, base=10), "neither"
    raise ValueError(f"unknown color_mode {color_mode!r}; expected 'clip', 'minmax', or 'symlog'")


def plot_bidstack_comparison(
    day: str,
    days_before: int = 7,
    units: list[str] | None = None,
    bidstack_df: pd.DataFrame | None = None,
    color_mode: str = "minmax",
    price_clip: float = 300,
    symlog_linthresh: float = 50,
    overlay_soc_and_price: bool = False,
):
    """6 rows (one per battery) x 2 columns (week-before | day) grid: each
    panel is that battery's bid stack across the day (see
    _draw_bidstack_panel) - x = time, bars stacked by tranche (cumulative
    MW), coloured by submitted price (turbo rainbow colormap - blue/green =
    low or negative price, yellow/red = high price), shared across the
    whole figure. Rainbow instead of a 2-colour diverging map so adjacent
    price bands stay visually distinct instead of everything mid-range
    collapsing toward the same pale colour. `color_mode` controls how price
    maps to colour:
      - "clip" (default): fixed +/-`price_clip` bounds: a handful of
        tranches sit at the AEMO offer price ceiling/floor (~+-$1000), and
        letting those set the scale washes out the range most bids
        actually live in, so anything beyond `price_clip` just saturates
        to full-intensity red/blue instead of stretching the scale.
      - "minmax": bounds are the actual +-max(abs(price)) in the figure -
        no clipping, but a single extreme bid can flatten everything else.
      - "symlog": symmetric log scale (linear within +-`symlog_linthresh`,
        log beyond) - compresses extreme values without hard-clipping them.
    Each row's y-axis (MW) is shared across its own two columns, like
    plot_day_comparison.

    Pass `bidstack_df` directly to bypass the normal parquet-cache loader -
    e.g. to test against a small hand-fetched sample before the full
    extraction has finished. Defaults to loading
    data/processed_data/bidstack.parquet."""
    units = units or battery_codes
    compare_day = (pd.Timestamp(day) - pd.Timedelta(days=days_before)).strftime("%Y-%m-%d")

    if bidstack_df is None:
        bidstack_df = _load_bidstack()
    if bidstack_df is None:
        return None

    day_data = bidstack_df[bidstack_df["dispatch_interval"].dt.strftime("%Y-%m-%d") == day]
    compare_data = bidstack_df[bidstack_df["dispatch_interval"].dt.strftime("%Y-%m-%d") == compare_day]

    if day_data.empty and compare_data.empty:
        print(f"{day}: no bidstack data for either {compare_day} or {day}, skipping plot_bidstack_comparison")
        return None

    norm, extend = _build_price_norm(color_mode, pd.concat([day_data, compare_data]), price_clip, symlog_linthresh)

    if overlay_soc_and_price:
        soc_df, _, _, price = _load_data()
        compare_soc, day_soc = _slice_day(soc_df, compare_day), _slice_day(soc_df, day)
        compare_price, day_price = _slice_day(price, compare_day), _slice_day(price, day)

    n = len(units)
    # the SOC/price overlay needs an offset third spine on the right of each
    # panel, so it gets extra right-margin and column spacing the plain
    # bidstack-only layout doesn't need
    gridspec_kw = (
        {"hspace": 0.15, "wspace": 0.35, "top": 0.92, "left": 0.06, "right": 0.78}
        if overlay_soc_and_price
        else {"hspace": 0.15, "wspace": 0.08, "top": 0.92, "right": 0.90}
    )
    fig, axes = plt.subplots(
        n,
        2,
        sharex="col",
        sharey="row",
        figsize=(22 if overlay_soc_and_price else 20, 2.3 * n),
        gridspec_kw=gridspec_kw,
        squeeze=False,
    )

    tz = None
    for i, code in enumerate(units):
        left_ax, right_ax = axes[i, 0], axes[i, 1]
        left_tz = _draw_bidstack_panel(left_ax, compare_data[compare_data["code"] == code], norm)
        right_tz = _draw_bidstack_panel(right_ax, day_data[day_data["code"] == code], norm)
        tz = tz or left_tz or right_tz

        left_ax.set_ylabel(f"{code}\nMW", fontsize="small")
        right_ax.set_ylabel("")
        plt.setp(right_ax.get_yticklabels(), visible=False)
        _style_axes(left_ax)
        _style_axes(right_ax)

        if overlay_soc_and_price:
            left_soc = compare_soc[code] if compare_soc is not None and code in compare_soc else None
            right_soc = day_soc[code] if day_soc is not None and code in day_soc else None
            # only the right (day) column carries axis numbers for the
            # overlay - both columns still draw the lines, this just halves
            # the tick clutter across a grid that's already dense
            _overlay_soc_and_price(left_ax, left_soc, compare_price, show_ticks=False)
            _overlay_soc_and_price(right_ax, right_soc, day_price, show_ticks=True)

    if tz is None:
        print(f"{day}: no data for either date after all, skipping plot_bidstack_comparison")
        plt.close(fig)
        return None

    compare_start = pd.Timestamp(compare_day, tz=tz)
    day_start = pd.Timestamp(day, tz=tz)

    axes[-1, 0].set_xlim(compare_start, compare_start + pd.Timedelta(days=1))
    axes[-1, 1].set_xlim(day_start, day_start + pd.Timedelta(days=1))

    for col in (0, 1):
        ax = axes[-1, col]
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))
        ax.set_xlabel("Time (AWST)")
        for row in range(n - 1):
            plt.setp(axes[row, col].get_xticklabels(), visible=False)

    axes[0, 0].set_title(f"{compare_day}  (week before)", fontsize="medium")
    axes[0, 1].set_title(f"{day}  (stress event)", fontsize="medium")

    cbar_labels = {
        "clip": f"Submitted price ($/MWh, clipped at +/-{price_clip:.0f})",
        "minmax": "Submitted price ($/MWh)",
        "symlog": f"Submitted price ($/MWh, symlog, linear within +/-{symlog_linthresh:.0f})",
    }

    sm = plt.cm.ScalarMappable(norm=norm, cmap="turbo")
    sm.set_array([])
    cbar_x = 0.97 if overlay_soc_and_price else 0.92
    cbar_ax = fig.add_axes((cbar_x, 0.15, 0.012, 0.7))
    fig.colorbar(sm, cax=cbar_ax, label=cbar_labels[color_mode], extend=extend)

    suptitle_y = 0.975
    if overlay_soc_and_price:
        overlay_handles = [
            plt.Line2D([0], [0], color=_SOC_LINE_COLOR, linewidth=2, label="SOC (MWh)"),
            plt.Line2D(
                [0], [0], color=_CLEARING_PRICE_LINE_COLOR, linewidth=1.8, linestyle="--", label="Clearing price ($/MWh)"
            ),
        ]
        fig.legend(
            handles=overlay_handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncols=2, fontsize="small", frameon=False
        )
        suptitle_y = 1.04

    fig.suptitle(f"Bid stack: {day} vs {days_before} days before", y=suptitle_y)
    fig.align_ylabels(list(axes[:, 0]))

    save_figure(fig, repo_plots_dir / "bidstack-comparison" / f"{day}_bidstack_vs_week_before_{color_mode}.png")
    return fig


def regenerate_new_plots():
    """Regenerate every plot this module produces: day (MWh + % variants),
    day-vs-week-before comparison (MWh + % variants), bidstack comparison,
    and headroom for each system_stress_events date, plus the whole-record
    entry-energy scatter. Bidstack comparison is skipped automatically
    (with a message) until bidstack.parquet exists."""
    from tools.constants import system_stress_events

    for event in system_stress_events:
        plot_day(event["date"], soc_unit="mwh")
        plot_day(event["date"], soc_unit="pct")
        plot_day_comparison(event["date"], soc_unit="mwh")
        plot_day_comparison(event["date"], soc_unit="pct")
        plot_bidstack_comparison(event["date"])
        plot_headroom(event["date"])

    plot_entry_energy()


if __name__ == "__main__":
    regenerate_new_plots()
