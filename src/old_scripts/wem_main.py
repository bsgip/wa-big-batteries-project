# Make sure you've installed with analysis extras
# uv add "openelectricity[analysis]"
from operator import index
import plotly.graph_objects as go
from scripts import wem
from wem_data.constants import *
from scripts.util import *
from scripts.defaults import esr_codes
from wem_data.paths import data_dir

start_time = "2023/10/1 00:00:00"
end_time = "2026/8/1 23:59:59"


battery_scada_data = wem.dynamic_data_compiler(start_time=start_time,
                                               end_time=end_time,
                                               table_name="facilityScada",
                                               filter_cols=["code"],
                                               filter_values=(esr_codes,),
                                               # rebuild=True,
                                               )

# battery_scada_data[WEMColumnName.dispatch_interval] = pd.to_datetime(
#     battery_scada_data[WEMColumnName.dispatch_interval], utc=True, errors='coerce'
#     )
# battery_scada_data[CustomColumnName.time_of_day] = (
#     battery_scada_data[WEMColumnName.dispatch_interval].dt.strftime('%H:%M')
#     )
# battery_scada_data[CustomColumnName.date_of_year] = (
#     battery_scada_data[WEMColumnName.dispatch_interval].dt.date
#     )
battery_scada_data.to_csv(data_dir / "wem_battery_scada.csv", index=False)


"""
for code in battery_codes:
    unit_scada_data = battery_scada_data[battery_scada_data[WEMColumnName.facility_code] == code]
    unit_scada_data = pd.pivot_table(unit_scada_data,
                                     values=WEMColumnName.quantity,
                                     index=[WEMColumnName.dispatch_interval,
                                            CustomColumnName.time_of_day,
                                            CustomColumnName.date_of_year],
                                     columns=[WEMColumnName.facility_code])
    unit_scada_data.reset_index(inplace=True)
"""