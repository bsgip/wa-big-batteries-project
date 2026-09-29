"""Turn extracted datasets into cleaned ones: reads
data/processed_data/extracted/, writes data/processed_data/clean/.

Cheap to re-run (seconds) - do it whenever cleaning logic or a constant it
depends on changes (a battery capacity, a sentinel value). Uncomment what you
want; fields with no cleaning step (demand, price, bidstack) aren't here at
all and get used straight from extracted/.

Usage:
    python -m data_processing.main
"""

import logging

from data_extraction.catalog import load
from data_processing import power, soc, unconstrained_forecast
from data_processing.store import save_clean

logger = logging.getLogger(__name__)


def main():
    save_clean("soc", soc.process(load("soc")))
    save_clean("power", power.process(load("power")))

    # The one field that fans out: one extracted frame in, three clean ones out
    # (clean/pv.parquet, clean/wind.parquet, clean/other_renewables.parquet).
    for name, frame in unconstrained_forecast.process(load("unconstrained_forecast")).items():
        save_clean(name, frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
