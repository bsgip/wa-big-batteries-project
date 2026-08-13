import pandas as pd
from utils.dirs import data_dir

pd.set_option("display.max_rows", None)
df = pd.read_csv(data_dir / "wem_battery_scada.csv")
