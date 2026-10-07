import logging

import pandas as pd

from data_extraction.catalog import load
from data_processing.demand import DEMAND_COLUMN, clean_demand
from data_processing.store import load_clean
from tools.constants import battery_codes
from visualisation.plot_distributions import plot_distribution_grid
from visualisation.plot_heatmap import plot_heatmap_grid


def main():
    pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
