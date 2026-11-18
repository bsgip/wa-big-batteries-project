"""Pre-dispatch price forecast error at fixed lead times (2 h, 6 h), monthly trend.

python -m exploration.forecast_error
"""

from analysis.price_forecast import errors_between, price_forecast_errors
from visualisation.plot_trend_band import trend_band_plot

LEAD_TIMES_H = [2, 4, 6]
PERIOD = ("2023-10-01", "2026-11-01")  # whole months only; Nov 2026 has two days


def main():
    errors = errors_between(price_forecast_errors(start=PERIOD[0]), *PERIOD)

    for lead in LEAD_TIMES_H:
        at_lead = errors.loc[errors["lead_time_h"] == lead].set_index("dispatch_interval")
        trend_band_plot(
            df=at_lead,
            col="error",
            show_outliers=False,
            ylabel="Forecast − actual price ($/MWh)",
            title=f"{lead}h-ahead price forecast error",
            filename=f"forecasts/price_forecast_error_{lead}h_month_trend.png",
        )


if __name__ == "__main__":
    main()
