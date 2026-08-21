from pathlib import Path

# Setup data directory
home_dir = Path.home()
data_dir = home_dir / "bigdata" / "wa-big-batteries-project"
raw_data_dir = data_dir / "raw_data"
processed_data_dir = data_dir / "processed_data"
plots_dir = data_dir / "plots"
case_input_dir = raw_data_dir / "caseInputData"
dispatch_solution_data_dir = raw_data_dir / "dispatchSolution" / "dispatchData"
predispatch_data_dir = raw_data_dir / "dispatchSolution" / "predispatchData"


def raw_dataset_dir(dataset: str) -> Path:
    """Where download.py and the parsers keep files for one dataset, e.g.
    "caseInputData" or "dispatchSolution/dispatchData". Holds only the
    downloaded/extracted .json files - nothing else."""
    return raw_data_dir / dataset


def download_state_dir(dataset: str) -> Path:
    """Where download.py tracks which zip archives it's already extracted for
    a dataset. Kept out of raw_dataset_dir so that directory stays pure data."""
    return data_dir / "download_state" / dataset
