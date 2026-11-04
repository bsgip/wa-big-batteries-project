"""Clean the extracted datasets: data/processed_data/extracted/ -> data/processed_data/clean/.
Takes seconds; uncomment what you want. Fields with no cleaning step (e.g. price) are used from extracted/.

    python -m data_processing.main
"""

import logging

from data_extraction.catalog import load
from data_processing import bidstack, power, soc, demand, dpv, rcoq, unconstrained_forecast
from tools.io import save_clean
from tools.registration import registered_capacity_mw

logger = logging.getLogger(__name__)


def main():
    # power tells SOC data gaps from outages for the fleet columns (see soc.add_fleet_soc_columns)
    save_clean("soc", soc.process(load("soc"), power=load("power")))
    save_clean("power", power.process(load("power")))

    # One extracted frame, three clean ones out (clean/demand, clean/underlying_demand, clean/dpv).
    dispatch_condition = load("dispatch_condition")
    save_clean("demand", demand.process(dispatch_condition))
    save_clean("underlying_demand", demand.process_underlying_demand(dispatch_condition))
    save_clean("dpv", dpv.process(dispatch_condition))

    save_clean("rcoq", rcoq.process(load("rcoq")))

    # One extracted frame in, three clean ones out
    # (clean/pv.parquet, clean/wind.parquet, clean/other_renewables.parquet).
    capacity = registered_capacity_mw()
    for name, frame in unconstrained_forecast.process(load("unconstrained_forecast"), capacity).items():
        save_clean(name, frame)

    save_clean("bidstack", bidstack.process(load("bidstack")))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
