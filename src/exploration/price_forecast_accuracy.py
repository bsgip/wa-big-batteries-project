"""How accurate are price forecasts at the horizons batteries plan over?
Pre-dispatch price forecast error by lead time (1-24 h), for midday and evening half-hours.

    python -m exploration.price_forecast_accuracy
"""

from analysis.price_forecast import error_by_lead_time, price_forecast_errors
from tools.constants import PEAK_ESROI_END, PEAK_ESROI_START
from visualisation.plot_price_forecast import plot_error_by_lead_time

LEAD_RANGE = (1, 24)  # hours
WINDOWS = {
    "Midday (09:00-15:00)": ("09:00", "15:00"),  # same as the daytime charging window
    f"Evening: ESROI ({PEAK_ESROI_START}-{PEAK_ESROI_END})": (PEAK_ESROI_START, PEAK_ESROI_END),
}


def main():
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


if __name__ == "__main__":
    main()
