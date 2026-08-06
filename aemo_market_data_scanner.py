#!/usr/bin/env python3
"""
aemo_market_data_scanner.py

Recursively walks the AEMO WA market-data directory listing
(https://data.wa.aemo.com.au/public/market-data/), downloads ONE example
file from each folder, and reports on its structure:

  - .zip   -> extracted, then each CSV/JSON inside is summarised
  - .csv   -> column headings listed
  - .json  -> top-level dictionary keys listed (or list info, if it's a list)

Usage:
    pip install requests beautifulsoup4
    python aemo_market_data_scanner.py

Notes:
  - This server is a classic IIS directory listing, so we parse the <a href>
    tags. Folders link to hrefs ending in "/", files don't.
  - Some of these folders contain years of dated files, so we take the FIRST
    file listed in each folder as the "exemplar" rather than downloading
    everything.
  - Be polite: this script throttles requests and skips already-visited
    folders. Increase MAX_DEPTH / remove it at your own risk - some branches
    (e.g. wemde) go quite deep and have a LOT of daily subfolders.
"""

import csv
import io
import json
import os
import re
import time
import zipfile
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT_URL = "https://data.wa.aemo.com.au/public/market-data/"
DOWNLOAD_DIR = "aemo_downloads"
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS = 0.5   # seconds, be polite to AEMO's server
MAX_DEPTH = 4                  # None = unlimited; some trees are very deep
MAX_FOLDERS = 200              # safety cap on total folders visited
MAX_DOWNLOAD_BYTES = 50 * 1024 * 1024  # skip exemplar files bigger than 50MB

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AEMO-DirScanner/1.0)"
}

session = requests.Session()
session.headers.update(HEADERS)


@dataclass
class Entry:
    name: str
    url: str
    is_dir: bool


@dataclass
class Stats:
    folders_visited: int = 0
    files_downloaded: int = 0
    errors: list = field(default_factory=list)


stats = Stats()


# ---------------------------------------------------------------------------
# Directory listing parsing
# ---------------------------------------------------------------------------

def list_directory(url: str) -> list[Entry]:
    """Fetch an IIS-style directory listing page and return its entries."""
    resp = session.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    entries = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        text = a.get_text(strip=True)

        if text == "[To Parent Directory]" or href in ("../", "/"):
            continue

        full_url = urljoin(url, href)

        # Don't wander outside the market-data tree
        if not full_url.startswith(ROOT_URL):
            continue

        is_dir = href.endswith("/")
        name = text or href.rstrip("/").split("/")[-1]
        entries.append(Entry(name=name, url=full_url, is_dir=is_dir))

    return entries


# ---------------------------------------------------------------------------
# File summarisation
# ---------------------------------------------------------------------------

def summarise_csv_bytes(data: bytes, label: str):
    text = data.decode("utf-8", errors="replace")
    reader = csv.reader(io.StringIO(text))
    try:
        headers = next(reader)
    except StopIteration:
        print(f"    [CSV] {label}: empty file")
        return
    print(f"    [CSV] {label}: {len(headers)} columns")
    for h in headers:
        print(f"        - {h}")


def summarise_json_bytes(data: bytes, label: str):
    try:
        obj = json.loads(data)
    except json.JSONDecodeError as e:
        print(f"    [JSON] {label}: could not parse ({e})")
        return

    if isinstance(obj, dict):
        keys = list(obj.keys())
        print(f"    [JSON] {label}: top-level dict with {len(keys)} keys")
        for k in keys:
            print(f"        - {k}")
    elif isinstance(obj, list):
        print(f"    [JSON] {label}: top-level list with {len(obj)} items")
        if obj and isinstance(obj[0], dict):
            print(f"        first item keys: {list(obj[0].keys())}")
    else:
        print(f"    [JSON] {label}: top-level type is {type(obj).__name__}")


def summarise_zip_bytes(data: bytes, label: str):
    try:
        zf = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile:
        print(f"    [ZIP] {label}: not a valid zip file")
        return

    names = zf.namelist()
    print(f"    [ZIP] {label}: {len(names)} file(s) inside")
    for name in names:
        lower = name.lower()
        if lower.endswith(".csv"):
            summarise_csv_bytes(zf.read(name), f"{label} -> {name}")
        elif lower.endswith(".json"):
            summarise_json_bytes(zf.read(name), f"{label} -> {name}")
        else:
            print(f"        (skipping non csv/json member: {name})")


def process_exemplar_file(entry: Entry):
    """Download one file and dispatch to the right summariser."""
    try:
        head = session.head(entry.url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        size = int(head.headers.get("Content-Length", 0) or 0)
        if size and size > MAX_DOWNLOAD_BYTES:
            print(f"    (skipping {entry.name}: {size/1e6:.1f} MB, over size limit)")
            return
    except requests.RequestException:
        pass  # HEAD not always supported - just try the GET

    try:
        resp = session.get(entry.url, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as e:
        msg = f"download failed for {entry.url}: {e}"
        stats.errors.append(msg)
        print(f"    ERROR: {msg}")
        return

    data = resp.content
    stats.files_downloaded += 1

    # Optionally save a local copy
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    local_path = os.path.join(DOWNLOAD_DIR, entry.name)
    with open(local_path, "wb") as f:
        f.write(data)

    lower = entry.name.lower()
    if lower.endswith(".zip"):
        summarise_zip_bytes(data, entry.name)
    elif lower.endswith(".csv"):
        summarise_csv_bytes(data, entry.name)
    elif lower.endswith(".json"):
        summarise_json_bytes(data, entry.name)
    else:
        print(f"    (file type not CSV/JSON/ZIP, skipping content summary: {entry.name})")


# ---------------------------------------------------------------------------
# Recursive walk
# ---------------------------------------------------------------------------

def walk(url: str, depth: int = 0):
    if MAX_FOLDERS and stats.folders_visited >= MAX_FOLDERS:
        return
    if MAX_DEPTH is not None and depth > MAX_DEPTH:
        return

    stats.folders_visited += 1
    indent = "  " * depth
    print(f"{indent}[DIR] {url}")

    try:
        entries = list_directory(url)
    except requests.RequestException as e:
        msg = f"listing failed for {url}: {e}"
        stats.errors.append(msg)
        print(f"{indent}  ERROR: {msg}")
        return

    time.sleep(SLEEP_BETWEEN_REQUESTS)

    subdirs = [e for e in entries if e.is_dir]
    files = [e for e in entries if not e.is_dir]

    if files:
        exemplar = files[0]
        print(f"{indent}  exemplar file: {exemplar.name}")
        process_exemplar_file(exemplar)
    else:
        print(f"{indent}  (no files in this folder)")

    for sub in subdirs:
        walk(sub.url, depth + 1)


def main():
    print(f"Starting scan at {ROOT_URL}")
    print(f"(max depth={MAX_DEPTH}, max folders={MAX_FOLDERS})\n")
    walk(ROOT_URL)

    print("\n--- Summary ---")
    print(f"Folders visited:   {stats.folders_visited}")
    print(f"Files downloaded:  {stats.files_downloaded}")
    print(f"Errors:            {len(stats.errors)}")
    for e in stats.errors:
        print(f"  - {e}")


if __name__ == "__main__":
    main()
