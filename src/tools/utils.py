import ijson
import pandas as pd
from pprint import pprint


def search_keyword(filepath, keyword: str):
    """Print every JSON path in filepath that contains keyword (lower-case match)."""
    hits = set()
    with open(filepath, "rb") as f:
        for prefix, event, value in ijson.parse(f):
            if keyword in prefix.lower():
                hits.add(prefix)
    pprint(hits)
