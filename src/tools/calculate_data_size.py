"""Recursively size up the entire AEMO wemde/ directory tree, printing one
line per directory (the total size of the files directly inside it, parsed
from the <pre> IIS listing) and a grand total at the end.

Usage:
    python -m wem_data.calculate_data_size
"""

import re

from bs4 import BeautifulSoup

from tools.constants import ROOT_URL
from wem_data.download import list_directory, session

# name, date, time+AM/PM, size (bytes) - fields are whitespace-separated in
# the IIS listing, but the padding between them varies, hence \s+ everywhere.
LISTING_ROW_RE = re.compile(
    r"(?P<name>\S+)\s+(?P<date>\d{1,2}/\d{1,2}/\d{4})\s+(?P<time>\d{1,2}:\d{2}\s*[AP]M)\s+(?P<size>\d+)"
)


def fetch_listing_text(url: str) -> str:
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.content, "html.parser")
    pre = soup.find("pre")
    if pre is None:
        raise ValueError(f"no <pre> block found at {url}")
    return pre.text


def parse_file_sizes(listing_text: str) -> dict[str, int]:
    """Return {filename: size_in_bytes} parsed from an IIS directory listing."""
    return {m["name"]: int(m["size"]) for m in LISTING_ROW_RE.finditer(listing_text)}


def walk_directory_sizes(url: str) -> int:
    """Recurse through the directory tree rooted at `url`, printing each
    directory's own file total, and return the grand total in bytes."""
    subdirs, _files = list_directory(url)
    own_total = sum(parse_file_sizes(fetch_listing_text(url)).values())

    label = url[len(ROOT_URL) :] or url
    print(f"{label}: {own_total:,} bytes ({own_total / 1e9:.2f} GB)")

    total = own_total
    for _name, suburl in subdirs:
        total += walk_directory_sizes(suburl)
    return total


if __name__ == "__main__":
    grand_total = walk_directory_sizes(ROOT_URL)
    print(f"\nTOTAL: {grand_total:,} bytes ({grand_total / 1e9:.2f} GB)")
