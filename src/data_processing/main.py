"""Turn extracted datasets into cleaned ones: reads
data/processed_data/extracted/, writes data/processed_data/clean/.

Cheap to re-run (seconds) - do it whenever cleaning logic or a constant it
depends on changes (a battery capacity, a sentinel value). Uncomment what you
want; fields with no cleaning step (price, bidstack) aren't here at all and
get used straight from extracted/.

Usage:
    python -m data_processing.main
"""

import logging

from data_extraction.catalog import load
from data_processing import power, soc, demand, dpv, rcoq, unconstrained_forecast
from tools.io import save_clean
from tools.registration import registered_capacity_mw

logger = logging.getLogger(__name__)


def main():
    # save_clean("soc", soc.process(load("soc")))
    # save_clean("power", power.process(load("power")))

    # One extracted frame, three clean ones out (clean/demand, clean/underlying_demand, clean/dpv).
    # dispatch_condition = load("dispatch_condition")
    # save_clean("demand", demand.process(dispatch_condition))
    # save_clean("underlying_demand", demand.process_underlying_demand(dispatch_condition))
    # save_clean("dpv", dpv.process(dispatch_condition))

    # The one field that fans out: one extracted frame in, three clean ones out
    # (clean/pv.parquet, clean/wind.parquet, clean/other_renewables.parquet).
    # save_clean("rcoq", rcoq.process(load("rcoq")))

    capacity = registered_capacity_mw()
    for name, frame in unconstrained_forecast.process(load("unconstrained_forecast"), capacity).items():
        save_clean(name, frame)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
