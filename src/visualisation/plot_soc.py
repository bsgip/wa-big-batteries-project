import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from tools.constants import battery_capacity_MWh, system_stress_events
from tools.paths import plots_dir
from wem_data.parse_energy_prices import build_energy_price_series_for_dates

battery_codes = list(battery_capacity_MWh.keys())


def _plot_event_day(
    df: pd.DataFrame,
    event: dict,
    columns: list[str],
    ylabel: str,
    title: str,
    filename: str,
    price: pd.Series | None = None,
):
    date = event["date"]
    day_df = df.loc[date]

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in columns:
        ax.plot(day_df.index, day_df[col], label=col.removesuffix("_soc_pct"))

    tz = day_df.index.tz

    ax.set_xlabel("Time (AWST)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2, tz=tz))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=tz))

    if event["event_start_time"] and event["event_end_time"]:
        event_start = pd.Timestamp(f"{date} {event['event_start_time']}", tz=tz)
        event_end = pd.Timestamp(f"{date} {event['event_end_time']}", tz=tz)
        ax.axvspan(event_start, event_end, color="gold", alpha=0.2, zorder=0, label="event window")

    if event["highest_stress_time"]:
        highest_stress_time = pd.Timestamp(f"{date} {event['highest_stress_time']}", tz=tz)
        ax.axvline(highest_stress_time, color="red", linewidth=1.5, zorder=0, label="highest stress time")

    handles, labels = ax.get_legend_handles_labels()

    if price is not None and not price.empty:
        price_ax = ax.twinx()
        (price_line,) = price_ax.plot(price.index, price.to_numpy(), "k--", linewidth=1, label="energy_price")
        price_ax.set_ylabel("Energy price ($/MWh)")
        handles.append(price_line)
        labels.append("energy_price")

    ax.legend(handles, labels, loc="upper left", fontsize="small")
    fig.autofmt_xdate()
    fig.tight_layout()

    plots_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(plots_dir / filename, dpi=200)
    plt.close(fig)


def plot_soc_for_stress_events(df: pd.DataFrame):
    """For each system stress event, save a battery SOC line plot in MWh and
    another in % (all batteries on one plot each), covering the full day,
    with the energy price overlaid as a dashed line."""
    pct_columns = [f"{code}_soc_pct" for code in battery_codes]

    event_dates = [event["date"] for event in system_stress_events]
    # one scan of the (huge) dispatchData folder for all event dates, instead
    # of one scan per event
    all_prices = build_energy_price_series_for_dates(event_dates)

    for event in system_stress_events:
        date = event["date"]
        price = all_prices.loc[date] if date in all_prices.index else None

        _plot_event_day(
            df,
            event,
            columns=battery_codes,
            ylabel="State of charge (MWh)",
            title=f"Battery SOC (MWh) — {date}",
            filename=f"soc_mwh_{date}.png",
            price=price,
        )

        _plot_event_day(
            df,
            event,
            columns=pct_columns,
            ylabel="State of charge (%)",
            title=f"Battery SOC (%) — {date}",
            filename=f"soc_pct_{date}.png",
            price=price,
        )
