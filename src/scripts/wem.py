from datetime import datetime
from scripts.custom_errors import UserInputError, NoDataToReturn, DataMismatchError
from scripts.util import *


def run(year, month, day, table_name, down_load_to=defaults.raw_data_cache):
    url_formatted_latest = defaults.dynamic_data_url_current.format(
        table_name, defaults.names_current[table_name], year, month, day
    )
    url_formatted_hist = defaults.dynamic_data_url_previous.format(
        table_name, defaults.names_previous[table_name], year, month, day
    )
    try:
        download_and_unpack_json_files(url_formatted_latest, down_load_to)
    except Exception as e:
        try:
            download_and_unpack_json_files(url_formatted_hist, down_load_to)
        except Exception as e:
            logger.warning(f"{url_formatted_hist} not downloaded")


def dynamic_data_compiler(
        start_time,
        end_time,
        table_name,
        raw_data_location=defaults.raw_data_cache,
        select_columns=None,
        filter_cols=None,
        filter_values=None,
        keep_csv=True,
        parse_data_types=True,
        rebuild=False, ):
    if not os.path.isdir(raw_data_location):
        raise UserInputError("The raw_data_location provided does not exist.")

    if table_name not in defaults.dynamic_tables:
        raise UserInputError("Table name provided is not a dynamic table.")

    logger.info(f"Compiling data for table {table_name}")

    start_time = datetime.strptime(start_time, "%Y/%m/%d %H:%M:%S")
    try:
        end_time = datetime.strptime(end_time, "%Y/%m/%d %H:%M:%S")
    except Exception as err:
        print(f"Unexpected {err=}, {type(err)=}")
        raise

    data_tables = _dynamic_data_fetch_loop(
        start_time=start_time,
        end_time=end_time,
        table_name=table_name,
        raw_data_location=raw_data_location,
        rebuild=rebuild,
        select_columns=select_columns)

    if data_tables:
        all_data = pd.concat(data_tables, sort=False)

        if parse_data_types:
            all_data = infer_column_data_types(all_data)

        if filter_cols is not None:
            all_data = filter_on_column_value(all_data, filter_cols, filter_values)

        logger.info(f"Returning {table_name}.")
        return all_data
    else:
        raise NoDataToReturn(
            (
                    f"Compiling data for table {table_name} failed. "
                    + "This probably because none of the requested data "
                    + "could be download from AEMO. Check your internet "
                    + "connection and that the requested data is archived on: "
                    + "https://data.wa.aemo.com.au/public/market-data/wemde see "
                    + "nemosis.defaults for table specific urls."
            )
        )


def _dynamic_data_fetch_loop(
        start_time,
        end_time,
        table_name,
        raw_data_location,
        select_columns=None,
        keep_csv=True,
        rebuild=False, ):
    date_gen = current_gen(start_time, end_time)

    data_tables = []
    for year, month, day, index in date_gen:
        file_name, path_name = create_filename(table_name=table_name,
                                               raw_data_location=raw_data_location,
                                               day=day, month=month, year=year)
        if not os.path.exists(path_name) or rebuild:
            run(year=year, month=month, day=day,
                table_name=table_name, down_load_to=raw_data_location)

        try:
            data = pd.read_csv(path_name)

            if select_columns is not None:
                data = filter_columns(data, table_name, select_columns, path_name)
            data_tables.append(data)
        except FileNotFoundError:
            pass

    return data_tables


def static_data_compiler():
    True
