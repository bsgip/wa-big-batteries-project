import logging

from analysis.extract_raw_data import extract_raw_data
from analysis.process_raw_data import process_raw_data
from tools.paths import repo_processed_data_dir
from visualisation.build_plots import build_plots

logger = logging.getLogger(__name__)

# One place to check at the end of a run: every step below reports here
# instead of letting an exception stop the rest of the pipeline.
_run_summary: list[str] = []


def _step(description: str, fn, *args, **kwargs):
    """Run one pipeline step. On failure, log it, record it in the
    end-of-run summary, and return None instead of raising - nothing here
    should ever be able to stop the rest of the pipeline from running."""
    try:
        result = fn(*args, **kwargs)
    except Exception as e:
        logger.error(f"FAILED: {description} ({type(e).__name__}: {e})")
        _run_summary.append(f"FAILED: {description} - {type(e).__name__}: {e}")
        return None
    _run_summary.append(f"OK: {description}")
    return result


def main():
    repo_processed_data_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: download/extract raw data
    raw_soc_df, demand, raw_power_df, price = extract_raw_data(_step)

    # Stage 2: process raw data
    soc_df, power_df = process_raw_data(_step, raw_soc_df, raw_power_df, demand, price)

    # Stage 3: plot
    build_plots(_step, soc_df, power_df, demand, price)

    logger.info("Done. Run summary:")
    for line in _run_summary:
        logger.info(f"  {line}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
