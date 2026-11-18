"""Which zero SOC readings does cleaning mask (outage, impossible drop) and which does it keep
(really empty)? See data_processing/soc.py for the rules.

    python -m exploration.soc
"""

import logging

from data_extraction.catalog import load
from data_processing.soc import clean_charge_level_sentinel_df, mask_unknown_zeros, offered_mw
from tools.constants import battery_codes
from tools.io import load_clean
from visualisation.plot_masked_zeros import plot_masked_zero_timeline


def main():
    raw = clean_charge_level_sentinel_df(load("soc")[battery_codes])
    offered = offered_mw(load_clean("bidstack", start_date=None), battery_codes)
    masked = mask_unknown_zeros(raw, offered)

    plot_masked_zero_timeline(raw, masked)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
