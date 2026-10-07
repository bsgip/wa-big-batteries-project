import pandas as pd
import matplotlib.pyplot as plt

from data_extraction.catalog import load
from visualisation.plot_trend_band import trend_band_event_plot


def daily_vol(s):
    """Daily price volatility measures from 5-min prices."""
    s = s.squeeze()              # one-column DataFrame -> Series
    r = s.resample('D')
    d = s.diff().abs().resample('D')
    out = pd.DataFrame({
        'median':   r.median(),
        'iqr':      r.quantile(.75) - r.quantile(.25),
        'range':    r.max() - r.min(),
        'mad':      d.mean(),
        'max_move': d.max(),
        'tv':       d.sum(),
        'n':        r.count(),
    })
    out['spike'] = r.max() - out['median']
    return out


def main():
    vol = daily_vol(load("price"))
    vol = vol[vol['n'] > 250]

    trend_band_event_plot(
        vol[['mad']],
        ylabel='Mean absolute 5-min price change ($/MWh)',
        title='Routine intraday price movement',
        filename='price_intraday_movement_trend_events.png',
    )
    plt.show()


if __name__ == "__main__":
    main()
