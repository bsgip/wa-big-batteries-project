"""4h-ahead pre-dispatch price forecast error, monthly trend.

    python -m exploration.forecast_error
"""

from analysis.price_forecast import price_forecast_errors
from visualisation.plot_trend_band import trend_band_plot


def main():
    errors = price_forecast_errors(start="2023-10-01")
    forecast_4h = errors.loc[errors["lead_time_h"] == 4].set_index("dispatch_interval")

    trend_band_plot(
        df=forecast_4h,
        col="error",
        show_outliers=False,
        ylabel="Forecast − actual price ($/MWh)",
        title="4h-ahead price forecast error",
        filename="price_forecast_error_4h_month_trend.png",
    )


if __name__ == "__main__":
    main()
