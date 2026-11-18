import datetime

import pandas as pd

from tools.constants import battery_capacity_MWh, battery_codes
from tools.io import load_clean

pd.set_option("display.max_columns", None)

fc = load_clean("fcess_bidstack")
print(fc[fc["notice_time"] > 1e2])
