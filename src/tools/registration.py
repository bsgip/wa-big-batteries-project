import pandas as pd

from tools.paths import registration_dir


facilities_path = registration_dir / "facilities.csv"
retirements_path = registration_dir / "facility_retirements.csv"  # written by data_processing.retirements

TIMEZONE = "Australia/Perth"  # as the market data's timestamps


def registered_capacity_mw(path=facilities_path):
    """{facility code: registered system size (MW)} from the AEMO facilities list, skipping blank sizes."""
    facilities = pd.read_csv(path)
    sized = facilities.dropna(subset=["System Size (MW)"])
    return dict(zip(sized["Facility Code"], sized["System Size (MW)"]))


def retirement_dates(path=retirements_path) -> dict[str, pd.Timestamp]:
    """{facility code: retirement date} from AEMO's retirement notices.

    The date is the expected closure, or the end of Reserve Capacity Obligations where there is
    no closure date. Each is midnight Perth time, so it compares directly with the data's index,
    e.g. power.index >= retirement_dates()["MUJA_G6"].

    Args:
        path: The clean CSV from data_processing.retirements.

    Returns:
        {facility code: timestamp}, earliest first.
    """
    retirements = pd.read_csv(path, parse_dates=["retirement_date"])
    dates = retirements["retirement_date"].dt.tz_localize(TIMEZONE)
    return dict(zip(retirements["facility_code"], dates))
