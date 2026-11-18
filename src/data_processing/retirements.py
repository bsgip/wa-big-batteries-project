"""AEMO's facility retirement notices (xlsx) -> data/registration/facility_retirements.csv.

The spreadsheet has one block of rows per facility: the first row holds the facility and its
dates, the rows under it only the Capacity Credits for later capacity years. The clean CSV
keeps one row per facility, with:

    expected_closure     when the facility closes (blank for demand side programmes)
    obligations_cease    when its Reserve Capacity Obligations end (blank for loads)
    retirement_date      expected_closure, or obligations_cease where there is no closure date

    python -m data_processing.retirements
"""

import logging

import pandas as pd

from tools.paths import registration_dir

logger = logging.getLogger(__name__)

raw_retirements_path = registration_dir / "Notifications of Facility retirements.xlsx"
retirements_path = registration_dir / "facility_retirements.csv"

HEADER_ROW = 2  # the column names are on the sheet's 3rd row
COLUMNS = {
    "Facility": "facility_code",
    "Market Participant": "participant",
    "Standing Data Name Plate Capacity (MW) ": "nameplate_mw",  # trailing space as in the sheet
    "Expected closure of Facility (4.4A.1.)": "expected_closure",
    "Reserve Capacity Obligations to Cease (4.4A.1A.)": "obligations_cease",
}


def read_notices(path=raw_retirements_path) -> pd.DataFrame:
    """The retirement notices spreadsheet as it is."""
    return pd.read_excel(path, header=HEADER_ROW)


def process(raw: pd.DataFrame) -> pd.DataFrame:
    """One row per retiring facility, with its closure and obligation dates.

    Args:
        raw: From read_notices.

    Returns:
        Columns facility_code, participant, nameplate_mw, expected_closure, obligations_cease
        and retirement_date, sorted by retirement_date.
    """
    df = raw[list(COLUMNS)].rename(columns=COLUMNS)

    # Only a block's first row names the facility; the footnotes under the table name none.
    df = df.dropna(subset=["facility_code"])
    df = df[df["facility_code"].str.match(r"^[A-Z0-9_]+\**$")]  # a code, maybe with footnote stars

    df["facility_code"] = df["facility_code"].str.rstrip("*").str.strip()
    df["participant"] = df["participant"].str.strip()
    df["retirement_date"] = df["expected_closure"].fillna(df["obligations_cease"])

    for column in ["expected_closure", "obligations_cease", "retirement_date"]:
        df[column] = df[column].dt.date  # dates only, so the CSV reads 2027-10-01

    return df.sort_values("retirement_date").reset_index(drop=True)


def main():
    retirements = process(read_notices())
    retirements.to_csv(retirements_path, index=False)
    logger.info(f"saved {len(retirements)} facilities to {retirements_path}")
    print(retirements.to_string())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
