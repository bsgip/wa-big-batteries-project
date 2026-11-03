"""Dispatch Solution JSON -> rows for the database. Never touches the database."""

from datetime import datetime


def run_from_file_name(file_name: str) -> dict[str, str | datetime]:
    """One raw file name -> the `runs` row it belongs to.

    e.g. "ReferenceDispatchSolution_202608121800.json" ->
        {"schedule": "Dispatch", "scenario": "Reference",
         "issued_at": datetime(2026, 8, 12, 18, 0, tzinfo=UTC+08:00), "file_variant": "regular"}

    Raises ValueError if the name isn't a Dispatch Solution file name.
    """
    raise NotImplementedError
