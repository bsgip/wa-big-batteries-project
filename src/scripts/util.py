from pathlib import Path
import json
import requests
import zipfile
import io
import os
import logging
import pandas as pd

from calendar import monthrange
from datetime import timedelta

from scripts.constants import *
from scripts import defaults

logger = logging.getLogger(__name__)

# Windows Chrome for User-Agent request headers
USR_AGENT_HEADER = {
    "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
            + " AppleWebKit/537.36 (KHTML, like Gecko) "
            + "Chrome/80.0.3987.87 Safari/537.36"
    )
}


def apply_dispatch_type_factor(dispatch_type):
    if DispatchType.load in str(dispatch_type).lower():
        return -1
    else:
        return 1


def write_plots(fig, plot_name, wem=False):
    Path("plots").mkdir(parents=True, exist_ok=True)
    if wem:
        plot_name = f"WEM/{plot_name}"
    fig.write_html("plots/{}.html".format(plot_name))


def get_time_of_day(date):
    return f"{date.hour}"


def get_date_of_year(date):
    return date.dayofyear


def get_day_of_week(date):
    return date.day_of_week


def get_week_of_year(date):
    return date.weekofyear


def download_and_unpack_json_files(url, down_load_to):
    """
    This function downloads a zipped json using a url,
    extracts the json and saves it a specified location
    """
    r = requests.get(url, headers=USR_AGENT_HEADER)
    if url.endswith("zip"):
        zipped_file = zipfile.ZipFile(io.BytesIO(r.content))
        file_name = zipped_file.namelist()[0]
        json_file = zipped_file.open(file_name).read()
    else:
        file_name = url.split("/")[-1]
        json_file = r.content
    json_data = json.loads(json_file)["data"]
    data = pd.DataFrame.from_dict(
        list(json_data.values())[0]
    )
    file_name = file_name.replace("json", "csv")
    data.to_csv(
        os.path.join(
            down_load_to,
            file_name
        ),
        index=False,
    )
    logger.info(f"{file_name} saved to csv.")


def current_gen(start_time, end_time):
    start_time = start_time - timedelta(days=1)

    end_year = end_time.year
    start_year = start_time.year

    for year in range(start_year, end_year + 1):

        if year == end_year:
            end_month = end_time.month
        else:
            end_month = 12

        if year == start_year:
            start_month = start_time.month - 1
        else:
            start_month = 0

        for month in defaults.months[start_month:end_month]:
            for day in range(1, monthrange(int(year), int(month))[1] + 1):
                if (
                        day < start_time.day
                        and int(month) == start_time.month
                        and year == start_year
                ) or (
                        day > end_time.day
                        and int(month) == end_time.month
                        and year == end_year
                ):
                    continue
                yield str(year), month, str(day).zfill(2), None


def download_csv(url, path_and_name):
    """
    This function downloads a zipped csv using a url,
    extracts the csv and saves it a specified location
    """
    r = requests.get(url, headers=USR_AGENT_HEADER)
    with open(path_and_name, "wb") as f:
        f.write(r.content)


def infer_column_data_types(data):
    """
    Infer datatype of DataFrame assuming inference need only be carried out
    for any columns with dtype "object". Adapted from StackOverflow.

    If the column is an object type, attempt conversions to (in order of):
    1. datetime
    2. numeric

    Returns: Data with inferred types.
    """

    def _get_series_type(series):
        if series.dtype == "object":
            try:
                col_new = pd.to_datetime(series)
                return col_new
            except Exception as e:
                try:
                    col_new = pd.to_numeric(series)
                    return col_new
                except Exception as e:
                    return series
        else:
            return series

    for col in data:
        series = data[col]
        typed = _get_series_type(series)
        data[col] = typed
    return data


def create_filename(table_name, raw_data_location, day, month, year):
    file_name = "{}_{}-{}-{}.csv".format(defaults.names_current[table_name],
                                         year, month, day)
    path_name = os.path.join(raw_data_location, file_name)
    return file_name, path_name


def filter_on_column_value(data, filter_cols, filter_values):
    for filter_col, filter_values in zip(filter_cols, filter_values):
        if filter_values is not None:
            data = data[data[filter_col].isin(filter_values)]
    return data


def filter_columns(data, table_name, select_columns, full_filename):
    headers = data.columns.tolist()
    columns = [
        column
        for column in defaults.table_columns[table_name]
        if column in headers
    ]

    available_cols = [c for c in columns if c in select_columns]
    if available_cols:
        data = data.loc[:, available_cols]

    rejected_cols = set(select_columns) - set(available_cols)
    if rejected_cols:
        logger.warning(
            f"{rejected_cols} not in {full_filename}. "
            + f"Loading {available_cols}"
        )

    return data

