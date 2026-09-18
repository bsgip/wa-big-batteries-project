"""Extract fields out of the raw WEMDE JSON into
data/processed_data/extracted/.

This is NOT a pipeline you run start to finish - it's a scratchpad. Uncomment
the line for whatever you actually want extracted and run it. Extraction is a
one-time job per field and an expensive one, so anything already extracted is
skipped unless you pass refresh=True.

Cleaning happens separately, in data_processing/main.py.

Usage:
    python -m data_extraction.main
"""

import logging

# extract is imported for the commented-out calls in main() - uncomment one and run.
from data_extraction.catalog import extract, list_datasets  # noqa: F401

logger = logging.getLogger(__name__)


def main():
    for name, description, done in list_datasets():
        print(f"  {name:9} {'[extracted]' if done else '[MISSING]  '}  {description}")

    # Fields from one corpus are grouped into a SINGLE walk, so ask for
    # everything you want from a source in one call rather than one at a time.

    extract(["soc", "bidstack"], refresh=True)   # one caseInputData walk (~1,056 files, ~70GB)
    # extract(["price", "power"])              # one dispatchSolution walk (~300k files, ~45min)

    # Explore a field on a couple of days first (dispatchSolution only).
    # Returns the frames for inspection without writing partial data to disk:
    # df = extract(["price"], dates=["2025-01-21"])["price"]

    # Re-extract something that already exists - only for a bug in the
    # extraction itself, not for a cleaning change:
    # extract(["soc"], refresh=True)

    # Registered in parse_case_input._FIELDS but never extracted, so asking
    # for it means a full caseInputData walk:
    # extract(["dispatch_condition"])   # needs a CATALOG entry first


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
