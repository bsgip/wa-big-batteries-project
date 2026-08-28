import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START, battery_codes, system_stress_events
from tools.paths import repo_plots_dir
from tools.plot_style import BATTERY_COLORS, save_figure


def _slice_day(data: pd.DataFrame | pd.Series | None, date: str):
    if data is None:
        return None
    try:
        day = data.loc[date]
    except KeyError:
        return None
    return None if day.empty else day


def _setup_event_axes(ax: plt.Axes, event: dict, tz):
    """Shared boilerplate for every event-day plot: x-axis formatting, the
    fixed 17:30-21:00 "Peak ESROI" shaded window, and the event's
    highest-stress-time marker (still per-event, unlike the shading)."""
    date = event["date"]

    ax.set_xlabel("Time (AWST)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))

    peak_start = pd.Timestamp(f"{date} {PEAK_ESROI_START}", tz=tz)
    peak_end = pd.Timestamp(f"{date} {PEAK_ESROI_END}", tz=tz)
    ax.axvspan(peak_start, peak_end, color="gold", alpha=0.2, zorder=0, label="Peak ESROI (17:30-21:00)")

    if event["highest_stress_time"]:
        highest_stress_time = pd.Timestamp(f"{date} {event['highest_stress_time']}", tz=tz)
        ax.axvline(highest_stress_time, color="red", linewidth=1.5, zorder=0, label="highest stress time")


def _finish_and_save(fig: plt.Figure, ax: plt.Axes, handles: list, labels: list, filename: str):
    ax.legend(handles, labels, loc="upper left", fontsize="small", ncols=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    save_figure(fig, repo_plots_dir / "events" / filename)


def _plot_metric_demand_price_event(
    metric_df_pct: pd.DataFrame,
    demand: pd.Series,
    price: pd.Series,
    event: dict,
    metric_label: str,
    metric_column_suffix: str,
    title_prefix: str,
    filename: str,
):
    date = event["date"]
    day_metric = _slice_day(metric_df_pct, date)
    if day_metric is None:
        print(f"{date}: no data for {metric_label}, skipping {filename}")
        return

    tz = day_metric.index.tz
    fig, ax = plt.subplots(figsize=(12, 6))

    for code in battery_codes:
        col = f"{code}{metric_column_suffix}"
        if col in day_metric:
            ax.plot(day_metric.index, day_metric[col], label=code, color=BATTERY_COLORS[code])
    ax.axhline(0, color="grey", linewidth=0.8, zorder=0)
    ax.set_ylabel(metric_label)
    ax.set_title(f"{title_prefix} — {date}")

    _setup_event_axes(ax, event, tz)
    handles, labels = ax.get_legend_handles_labels()

    day_demand = _slice_day(demand, date)
    if day_demand is not None:
        ax_demand = ax.twinx()
        (demand_line,) = ax_demand.plot(
            day_demand.index, day_demand.to_numpy(), "k--", linewidth=1, label="demand"
        )
        ax_demand.set_ylabel("Demand (MW)")
        handles.append(demand_line)
        labels.append("demand")

    day_price = _slice_day(price, date)
    if day_price is not None:
        ax_price = ax.twinx()
        ax_price.spines["right"].set_position(("outward", 60))
        (price_line,) = ax_price.plot(
            day_price.index, day_price.to_numpy(), color="grey", linestyle=":", linewidth=1.2, label="energy price"
        )
        ax_price.set_ylabel("Energy price ($/MWh)")
        handles.append(price_line)
        labels.append("energy price")

    _finish_and_save(fig, ax, handles, labels, filename)


def plot_power_demand_price_event(power_pct_df: pd.DataFrame, demand: pd.Series, price: pd.Series, event: dict):
    _plot_metric_demand_price_event(
        power_pct_df,
        demand,
        price,
        event,
        metric_label="Power (% of rated capacity, + discharge / - charge)",
        metric_column_suffix="_power_pct",
        title_prefix="Battery power vs demand vs price",
        filename=f"{event['date']}_power_demand_price.png",
    )


def plot_soc_demand_price_event(soc_pct_df: pd.DataFrame, demand: pd.Series, price: pd.Series, event: dict):
    _plot_metric_demand_price_event(
        soc_pct_df,
        demand,
        price,
        event,
        metric_label="State of charge (%)",
        metric_column_suffix="_soc_pct",
        title_prefix="Battery SOC vs demand vs price",
        filename=f"{event['date']}_soc_demand_price.png",
    )


def plot_soc_mwh_demand_price_event(soc_mwh_df: pd.DataFrame, demand: pd.Series, price: pd.Series, event: dict):
    """Same as plot_soc_demand_price_event, but SOC in MWh instead of % of
    (empirically-derived) rated capacity."""
    _plot_metric_demand_price_event(
        soc_mwh_df,
        demand,
        price,
        event,
        metric_label="State of charge (MWh)",
        metric_column_suffix="",
        title_prefix="Battery SOC (MWh) vs demand vs price",
        filename=f"{event['date']}_soc_mwh_demand_price.png",
    )


def plot_soc_power_event(soc_mwh_df: pd.DataFrame, power_mw_df: pd.DataFrame, event: dict):
    """SOC (MWh, left) vs power (MW, right) for every battery - solid line
    is SOC, dashed line is power, same colour per battery."""
    date = event["date"]
    day_soc = _slice_day(soc_mwh_df, date)
    day_power = _slice_day(power_mw_df, date)
    if day_soc is None:
        print(f"{date}: no SOC data, skipping soc_power plot")
        return

    tz = day_soc.index.tz
    fig, ax_soc = plt.subplots(figsize=(12, 6))
    ax_power = ax_soc.twinx()

    for code in battery_codes:
        if code in day_soc:
            ax_soc.plot(day_soc.index, day_soc[code], label=f"{code} (SOC)", color=BATTERY_COLORS[code], linestyle="-")
        if day_power is not None and code in day_power:
            ax_power.plot(
                day_power.index, day_power[code], label=f"{code} (power)", color=BATTERY_COLORS[code], linestyle="--"
            )

    ax_power.axhline(0, color="grey", linewidth=0.8, zorder=0)
    ax_soc.set_ylabel("State of charge (MWh)")
    ax_power.set_ylabel("Power (MW, + discharge / - charge)")
    ax_soc.set_title(f"Battery SOC vs power — {date}")

    _setup_event_axes(ax_soc, event, tz)
    handles, labels = ax_soc.get_legend_handles_labels()
    power_handles, power_labels = ax_power.get_legend_handles_labels()
    handles += power_handles
    labels += power_labels

    _finish_and_save(fig, ax_soc, handles, labels, f"{date}_soc_power.png")


def plot_event_day_summaries(soc_df: pd.DataFrame, power_df: pd.DataFrame, demand: pd.Series, price: pd.Series):
    """For each system stress event, save four plots covering all
    batteries: power vs demand vs price, SOC (%) vs demand vs price, SOC
    (MWh) vs demand vs price, and SOC vs power - each with the fixed
    17:30-21:00 "Peak ESROI" shaded window and the event's
    highest-stress-time marker."""
    for event in system_stress_events:
        plot_power_demand_price_event(power_df, demand, price, event)
        plot_soc_demand_price_event(soc_df, demand, price, event)
        plot_soc_mwh_demand_price_event(soc_df, demand, price, event)
        plot_soc_power_event(soc_df, power_df, event)
