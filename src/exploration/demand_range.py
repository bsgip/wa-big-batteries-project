"""Daily demand range (max - min, MW) -> clean/demand_range.parquet, and SOC plotted against it.

    python -m exploration.demand_range
"""

import logging

from analysis.demand_measures import broadcast_to_intervals, daily_demand_range
from tools.io import load_clean, save_clean
from visualisation.plot_soc_correlations import plot_soc_vs


def main():
    demand = load_clean("demand")["demand"]

    demand_range = daily_demand_range(demand)
    save_clean("demand_range", demand_range)

    soc = load_clean("soc")
    plot_soc_vs(
        soc_df=soc,
        x_series=broadcast_to_intervals(demand_range["range"], soc.index),
        x_label="Daily demand range (MW)",
        title="Battery SOC vs daily demand range - full history",
        filename="soc_vs_demand_range.png",
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
