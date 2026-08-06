import duckdb
import ijson
from scripts.util import download_and_unpack_json_files
from utils.downloads import download_file
from utils.urls import case_input_current_url
from utils.dirs import data_dir


# Download files
filepath = data_dir / "case_input.json"
# download_file(case_input_current_url, filepath)

duckdb.sql(f"DESCRIBE SELECT * FROM read_json_auto('{filepath}')").show()

with open(filepath, 'rb') as f:
    for rec in ijson.items(f, 'data.caseData.item', use_float=True):
        for key, val in rec.items():
            if isinstance(val, list):
                print(f"{key}: LIST of {len(val)} — first item: {val[0]}")
            else:
                print(f"{key}: {val!r}")
        break