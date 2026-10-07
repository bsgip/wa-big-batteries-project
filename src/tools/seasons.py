import pandas as pd


# Southern-hemisphere seasons, in calendar order starting from summer
SEASONS = ["Summer", "Autumn", "Winter", "Spring"]

SEASON_BY_MONTH = {
    12: "Summer", 1: "Summer", 2: "Summer",
    3: "Autumn", 4: "Autumn", 5: "Autumn",
    6: "Winter", 7: "Winter", 8: "Winter",
    9: "Spring", 10: "Spring", 11: "Spring",
}


def season_of(index):
    """Season name for each timestamp in index."""
    return pd.Index(index.month.map(SEASON_BY_MONTH), name="season")


def season_year_of(index):
    """Year each timestamp's season belongs to; December counts toward the next year's summer."""
    return pd.Index(index.year + (index.month == 12), name="season_year")
