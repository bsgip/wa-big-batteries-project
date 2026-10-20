import pandas as pd
import matplotlib.pyplot as plt
from visualisation.plot_time_of_day import plot_timeofday_box
from visualisation.plot_distributions import _plot_one_distribution
from tools.paths import extracted_data_dir, local_plots_dir
from tools.constants import battery_capacity_MWh
from tools.io import load_clean


pd.set_option("display.max_columns", None)


