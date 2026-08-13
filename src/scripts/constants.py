import pandas as pd


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
