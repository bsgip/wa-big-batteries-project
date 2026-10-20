import pandas as pd

from tools.paths import registration_dir


facilities_path = registration_dir / "facilities.csv"


def registered_capacity_mw(path=facilities_path):
    """{facility code: registered system size (MW)} from the AEMO facilities list, skipping blank sizes."""
    facilities = pd.read_csv(path)
    sized = facilities.dropna(subset=["System Size (MW)"])
    return dict(zip(sized["Facility Code"], sized["System Size (MW)"]))
