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
import pandas as pd

from main import _extract_power_and_price, _extract_soc_and_demand
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, battery_capacity_MW, battery_codes
from tools.df_management import clean_charge_level_df, derive_capacity_from_observed_max
from tools.paths import repo_plots_dir
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

    raw_soc_df, demand_df = _extract_soc_and_demand()
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


def regenerate_new_plots():
    """Regenerate every plot this module produces: day (MWh + % variants),
    day-vs-week-before comparison (MWh + % variants), and headroom for each
    system_stress_events date, plus the whole-record entry-energy scatter."""
    from tools.constants import system_stress_events

    for event in system_stress_events:
        plot_day(event["date"], soc_unit="mwh")
        plot_day(event["date"], soc_unit="pct")
        plot_day_comparison(event["date"], soc_unit="mwh")
        plot_day_comparison(event["date"], soc_unit="pct")
        plot_headroom(event["date"])

    plot_entry_energy()


if __name__ == "__main__":
    regenerate_new_plots()
