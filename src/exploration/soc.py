"""Where do the sustained runs of zero SOC that processing masks out fall in the timeline?

    python -m exploration.soc
"""

import logging

import pandas as pd

from data_processing.soc import mask_sustained_zero_runs
from tools.paths import local_processed_data_dir
from visualisation.plot_sustained_zeros import plot_sustained_zero_timeline


def main():
    df = pd.read_parquet(local_processed_data_dir / "soc.parquet")
    print((df == 0).sum())

    masked = mask_sustained_zero_runs(df, 60)
    print((masked == 0).sum())

    plot_sustained_zero_timeline(df, masked)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
