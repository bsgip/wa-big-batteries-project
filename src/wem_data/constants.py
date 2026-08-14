import pandas as pd
import pandas as pd
from wem_data.dirs import data_dir


class WEMColumnName:
    quantity = "quantity"
    facility_code = "code"
    dispatch_interval = "dispatchInterval"


class CustomColumnName:
    time_of_day = "Time of day"
    date_of_year = "Date of year"
    multiplier = "multiplier"
    minutes_of_day = "Minute of day"
    day_of_year = "Day of year"
    day_of_week = "Day of week"
    week_of_year = "Week of year"
    adjusted_initial_mw = "adjusted_INITIALMW"


class DispatchType:
    bidirectional = "bidirectional"
    generating = "generating"
    load = "load"


class VarNum:
    mw = 1
    gen_mw = 2
    gem_reg_com_mw = 5


axis_date_range_slider = dict(
    rangeselector=dict(
        buttons=list([
            {'count': 1, 'label': "1d", 'step': "day", 'stepmode': "backward"},
            dict(count=7,
                 label="1w",
                 step="day",
                 stepmode="backward"),
            dict(count=1,
                 label="1m",
                 step="month",
                 stepmode="backward"),
            dict(count=1,
                 label="YTD",
                 step="year",
                 stepmode="todate"),
            dict(count=1,
                 label="1y",
                 step="year",
                 stepmode="backward"),
            dict(step="all")
        ])
    ),
    rangeslider=dict(
        visible=True,
        thickness=0.05,
        yaxis=dict(rangemode='match'),
    ),
    type="date"
)


# AEMO list of facilities URL
facilities_url = "https://data.wa.aemo.com.au/datafiles/post-facilities/facilities.csv"
case_input_current_url = "https://data.wa.aemo.com.au/public/market-data/wemde/caseInputData/current/ReferenceDispatchCase_202608010800.json"



raw_data_cache = data_dir

dynamic_data_url_current = "https://data.wa.aemo.com.au/public/market-data/wemde/{}/current/{}_{}-{}-{}.json"
dynamic_data_url_previous = "https://data.wa.aemo.com.au/public/market-data/wemde/{}/previous/{}_{}{}{}.zip"

names_current = {
    "facilityScada": "SCADA"
}

names_previous = {
    "facilityScada": "FacilityScada"
}

primary_date_columns = {
    "facilityScada": "dispatchInterval",
}

months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]

dynamic_tables = [
    "facilityScada"
]

infographic_url = "https://data.wa.aemo.com.au/public/infographic/{}.csv"

infographic_tables = [
    "facility-meta",
    "facility-meta-renewables",
    "facility-meta-fuelmix",
    "generation",
    "participant"
]

table_columns = {
    "facilityScada": [
        "dispatchInterval",
        "code",
        "quantity"
    ],
    "facility-meta": [
        "PARTICIPANT_CODE",
        "FACILITY_CODE",
        "DISPLAY_NAME",
        "FACILITY_TYPE",
        "PRIMARY_FUEL",
        "ALTERNATE_FUEL",
        "GENERATION_TYPE",
        "YEAR_COMMISSIONED",
        "REGISTRATION_DATE",
        "CAPACITY_CREDITS",
        "RAMP_UP",
        "RAMP_DOWN",
        "LONGITUDE",
        "LATITUDE",
        "AS_AT"
    ],
    "participant":[
        "Participant Code",
        "Participant Name",
        "Address",
        "City",
        "State",
        "Postcode",
        "Country",
        "Phone",
        "Fax",
        "Contact Name",
        "Contact Position",
        "AEMO",
        "Market Participant",
        "Network Operator",
        "Extracted At"
    ]

}



esr_codes = sorted([
    "ALINTA_WGP_ESR1",
    "COLLIE_BESS2",
    "COLLIE_ESR1",
    "COLLIE_ESR4",
    "COLLIE_ESR5",
    "KWINANA_ESR1",
    "KWINANA_ESR2",
    "SBSOLAR1_CUNDERDIN_PV1",
    "TESLA_PICTON_G1",
    "PRDSO_WALPOLE_HG1",  # fuel type: hydro
])


# this information is obtained from the internet
battery_capacity_MWh = {
    "ALINTA_WGP_ESR1": 200,
    "COLLIE_BESS2": 2400,
    "COLLIE_ESR1": 800,
    "COLLIE_ESR4": 2000,
    "COLLIE_ESR5": 4000,
    "KWINANA_ESR1": 200,
    "KWINANA_ESR2": 900,
}