"""Pre-dispatch price forecast error by lead time (1-24 h): midday vs evening, year by year, and both.

    python -m exploration.price_forecast_accuracy
"""

from analysis.price_forecast import (
    error_by_lead_time,
    errors_between,
    errors_on_days,
    highest_days,
    price_forecast_errors,
)
from analysis.price_measures import daily_price_measures
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START
from tools.io import load_clean
from visualisation.plot_price_forecast import plot_error_by_lead_time, plot_windows_by_lead_time

LEAD_RANGE = (1, 24)  # hours
WINDOWS = {
    "Midday (09:00-15:00)": ("09:00", "15:00"),  # same as the daytime charging window
    f"Evening: ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})": (PEAK_ESROI_START, PEAK_ESROI_END),
}
WINDOW_SLUGS = dict(zip(WINDOWS, ["midday", "esroi"]))  # for filenames
WHOLE_DAY = ("00:00", "24:00")
WINDOW_COLORS = dict(zip(WINDOWS, ["0.6", "#D55E00"]))  # as in the offer plots: outside / in ESROI
TOP_SHARE = 0.10  # highest-price evenings: top 10% of each year by mean ESROI price
YEARS = {  # Oct-Sep, so the first year starts with the data (late Sep 2023)
    "Oct 2023 - Sep 2024": ("2023-10-01", "2024-10-01"),
    "Oct 2024 - Sep 2025": ("2024-10-01", "2025-10-01"),
    "Oct 2025 - Sep 2026": ("2025-10-01", "2026-10-01"),
}


def main():
    # Midday vs evening, over the analysis period
    errors = price_forecast_errors()  # analysis period only (from START_DATE)
    period = f"{errors['dispatch_interval'].min():%b %Y} - {errors['dispatch_interval'].max():%b %Y}"

    stats = {label: error_by_lead_time(errors, window, LEAD_RANGE) for label, window in WINDOWS.items()}
    plot_error_by_lead_time(
        stats,
        title=f"Pre-dispatch price forecast error by lead time ({period})",
        filename="forecasts/price_forecast_error_by_lead_time.png",
    )
    for label, m in stats.items():
        print(f"\n{label}\n{m.loc[[1, 2, 4, 8, 12, 16, 20, 24]].round(1)}")

    # Year by year, every half-hour of the day
    all_errors = price_forecast_errors(start=YEARS["Oct 2023 - Sep 2024"][0])
    by_year = {
        label: error_by_lead_time(errors_between(all_errors, start, end), WHOLE_DAY, LEAD_RANGE)
        for label, (start, end) in YEARS.items()
    }
    plot_error_by_lead_time(
        by_year,
        title="Pre-dispatch price forecast error by lead time, by year",
        filename="forecasts/price_forecast_error_by_lead_time_by_year.png",
    )
    for label, m in by_year.items():
        print(f"\n{label}\n{m.loc[[1, 2, 6, 12, 24], ['median', 'p10', 'p90', 'median_abs', 'forecasts']].round(1)}")

    # One window at a time (midday, then ESROI), year by year
    for label, window in WINDOWS.items():
        window_by_year = {
            year: error_by_lead_time(errors_between(all_errors, start, end), window, LEAD_RANGE)
            for year, (start, end) in YEARS.items()
        }
        plot_error_by_lead_time(
            window_by_year,
            title=f"Pre-dispatch price forecast error by lead time, {label}, by year",
            filename=f"forecasts/price_forecast_error_by_lead_time_{WINDOW_SLUGS[label]}_by_year.png",
        )

    # Midday vs evening, year by year
    by_year_window = {
        year: {label: error_by_lead_time(errors_between(all_errors, start, end), window, LEAD_RANGE)
               for label, window in WINDOWS.items()}
        for year, (start, end) in YEARS.items()
    }
    plot_windows_by_lead_time(
        by_year_window,
        column="median_abs",
        colors=WINDOW_COLORS,
        ylabel="Typical miss: median |forecast − actual| ($/MWh)",
        title="Pre-dispatch price forecast error by lead time: midday vs evening, by year",
        filename="forecasts/price_forecast_error_midday_vs_evening_by_year.png",
    )

    # ESROI, only the highest-price evenings of each year
    esroi = (PEAK_ESROI_START, PEAK_ESROI_END)
    esroi_price = daily_price_measures(load_clean("price", start_date=None)["energy_price"], *esroi, min_intervals=None)["mean"]
    top_by_year = {}
    for year, (start, end) in YEARS.items():
        days = highest_days(esroi_price, start, end, TOP_SHARE)
        print(f"{year}: {len(days)} evenings, mean ESROI price >= ${esroi_price.loc[days].min():,.0f}/MWh")
        top_by_year[year] = error_by_lead_time(errors_on_days(all_errors, days), esroi, LEAD_RANGE)
    plot_error_by_lead_time(
        top_by_year,
        title=f"Pre-dispatch price forecast error by lead time, ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END}), "
              f"top {TOP_SHARE:.0%} highest-price evenings of each year",
        filename="forecasts/price_forecast_error_by_lead_time_esroi_top_price_by_year.png",
    )
    for year, m in top_by_year.items():
        print(f"\n{year}\n{m.loc[[1, 2, 6, 12, 24], ['p10', 'median', 'p90', 'forecasts']].round(0)}")


if __name__ == "__main__":
    main()
