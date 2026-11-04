"""Extract fields from the raw WEMDE JSON into data/processed_data/extracted/.
Existing datasets are skipped unless refresh=True.

    python -m data_extraction.main
"""

import logging

from data_extraction.catalog import extract, list_datasets

logger = logging.getLogger(__name__)


def main():
    for name, description, done in list_datasets():
        print(f"  {name:40} {'[extracted]' if done else '[MISSING]  '}  {description}")

    # Case input run
    extract(
        [
            "soc",
            "bidstack",
            "dispatch_condition",
            "unconstrained_forecast",
            "power",
        ],
        refresh=True,
    )

    extract(["fcess_bidstack"], refresh=True)

    # Dispatch solution run
    extract(
        [
            "price",
            "price_setting",
            "market_shortfalls",
            "market_service_requirements",
            "dispatch_total",
            "available_quantities",
            "in_service_quantities",
            "projected_level",
            "energy_schedule",
            "fcess_schedule",
        ],
        refresh=True,
    )

    # Predispatch solution run
    extract(
        [
            "predispatch_price",
            "predispatch_price_setting",
            "predispatch_market_shortfalls",
            "predispatch_market_service_requirements",
            "predispatch_dispatch_total",
            "predispatch_projected_level",
            "predispatch_energy_schedule",
        ],
        refresh=True,
    )

    # RCOQ, from the monthly CSVs in data/rcoq/
    # extract(["rcoq"], refresh=True)

    # Explore a field on a few days without saving
    # df = extract(["price"], dates=["2025-01-21"])["price"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
