"""Build a self-contained HTML copy of a report notebook, for GitHub Pages.

Manual step, run by hand whenever a report is ready to (re)publish - no CI,
no watching for pushes. Markdown image links in the report notebooks (e.g.
`![](../../data/plots/2025-08-25_day.png)`) point at data/, which is
gitignored, so a plain `jupyter nbconvert --to html` would keep those same
relative links and break once the output is served from docs/ on its own.
This embeds every local image as a base64 data URI first, so the resulting
HTML has no dependency on data/ existing at all - copy/paste-safe, and safe
to commit to docs/ for Pages to serve.

The page itself is our own template, not nbconvert's JupyterLab one:
nbconvert's "basic" template renders just the cells, and this module wraps
them in a page with a collapsible sidebar table of contents (built from the
markdown headings), a sticky bar showing the current section, and styling
for figures and tables.

Header: the notebook's first cell becomes the page header (eyebrow line,
title, one-line summary, fact pills) - see pop_header() for the format. An
"Updated <date>" pill is added automatically.

Usage:
    uv run python -m reports.publish [notebook_path] [output_path]

Defaults to report-v{report_version}.ipynb -> docs/index.html.
"""

import base64
import datetime as dt
import html
import re
import sys
from pathlib import Path

import mistune
import nbformat
from bs4 import BeautifulSoup, NavigableString
from nbconvert import HTMLExporter

from tools.paths import local_docs_dir

report_version = 4
DEFAULT_NOTEBOOK = Path(__file__).parent / f"report-v{report_version}.ipynb"
DEFAULT_OUTPUT = local_docs_dir / "index.html"

_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".gif": "image/gif",
}

_IMG_LINK = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")


# --------------------------------------------------------------------------
# Images
# --------------------------------------------------------------------------


def _embed_one_image(match: re.Match, notebook_dir: Path) -> str:
    alt, src = match.group(1), match.group(2)
    if src.startswith(("http://", "https://", "data:")):
        return match.group(0)

    image_path = (notebook_dir / src).resolve()
    mime = _MIME_TYPES.get(image_path.suffix.lower())
    if mime is None or not image_path.exists():
        raise FileNotFoundError(f"can't embed image referenced as {src!r} - not found at {image_path}")

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"![{alt}](data:{mime};base64,{encoded})"


def embed_local_images(nb, notebook_dir: Path) -> None:
    """Rewrite every markdown-cell image link that points at a local file
    into a base64 data URI, in place. http(s) links and existing data URIs
    are left untouched."""
    for cell in nb.cells:
        if cell.cell_type == "markdown":
            cell.source = _IMG_LINK.sub(lambda m: _embed_one_image(m, notebook_dir), cell.source)


# --------------------------------------------------------------------------
# Header (title cell)
# --------------------------------------------------------------------------

_HEADING_LINE = re.compile(r"^#{1,6}\s+(.+?)\s*#*$")
_QUOTE_LINE = re.compile(r"^>\s?(.*)$")
_BULLET_LINE = re.compile(r"^[-*+]\s+(.+)$")


def _inline_md(text: str) -> str:
    """Render one line of markdown (bold, links, code...) without the <p>."""
    out = mistune.html(text).strip()
    return out[3:-4] if out.startswith("<p>") and out.endswith("</p>") else out


def pop_header(nb, fallback_title: str) -> dict:
    """Read the page header from the notebook's first cell, and remove that
    cell so it isn't rendered again as part of the report. The cell may hold,
    in this order (everything except the heading is optional):

        ANU × AEMO WA · Storage behaviour statistics (D2)   <- eyebrow
        # Report title                                      <- title
        > One-sentence summary of what the report answers.  <- lede
        - Data: Dec 2024 – Aug 2026                         <- pills
        - Draft

    If the first cell has anything else in it, it's left alone and the title
    falls back to notebook metadata "title", then the file name."""
    header = {"eyebrow": "", "title": nb.metadata.get("title") or fallback_title, "lede": "", "pills": []}
    if not nb.cells or nb.cells[0].cell_type != "markdown":
        return header

    eyebrow, lede, pills, title = [], [], [], None
    for line in (ln.strip() for ln in nb.cells[0].source.splitlines()):
        if not line:
            continue
        if title is None and (m := _HEADING_LINE.match(line)):
            title = m.group(1)
        elif title is None:
            eyebrow.append(line)
        elif m := _QUOTE_LINE.match(line):
            lede.append(m.group(1))
        elif m := _BULLET_LINE.match(line):
            pills.append(m.group(1))
        else:
            return header  # ordinary content: not a header cell

    if title is None:
        return header
    nb.cells.pop(0)
    return {
        "eyebrow": _inline_md(" ".join(eyebrow)) if eyebrow else "",
        "title": title.replace("**", "").replace("__", "").strip(),
        "lede": _inline_md(" ".join(lede)) if lede else "",
        "pills": [_inline_md(p) for p in pills],
    }


# --------------------------------------------------------------------------
# Body post-processing: figures, tables, headings -> contents tree
# --------------------------------------------------------------------------

# How many heading levels get automatic numbers: 1 = sections only
# ("Section 1"), 2 = subsections too (1.1, 1.2 ...).
NUMBER_DEPTH = 1

_UNNUMBERED = re.compile(r"\s*\{(?:-|\.unnumbered)\}\s*$")
_NUM_PREFIX = re.compile(r"^\s*\d+(?:\.\d+)*\.?\s+")
_NUMERIC = re.compile(r"^[-+−]?[\d,]*\.?\d+\s*%?$")


def _is_blank(node) -> bool:
    return (isinstance(node, NavigableString) and not node.strip()) or getattr(node, "name", None) == "br"


def _heading_label(h) -> str:
    return "".join(
        s for s in h.find_all(string=True) if not (s.parent.name == "a" and "anchor-link" in s.parent.get("class", []))
    ).strip()


def enhance_body(body: str) -> tuple[str, list[dict]]:
    """Restyle nbconvert's cell HTML and collect the headings.

    Returns (body_html, contents_tree)."""
    soup = BeautifulSoup(body, "html.parser")

    # Code cells with no output (inputs are excluded) render as empty boxes.
    for cell in soup.select("div.code_cell"):
        if not cell.select(".output_area"):
            cell.decompose()

    # A markdown paragraph holding only images becomes a figure card.
    for p in soup.select(".text_cell_render p"):
        kids = [c for c in p.contents if not _is_blank(c)]
        if kids and all(getattr(c, "name", None) == "img" for c in kids):
            for c in list(p.contents):
                if _is_blank(c):
                    c.extract()
            p.name = "figure"
            p["class"] = ["fig"]
    for img in soup.select(".fig img"):
        img["loading"] = "lazy"

    # Tables: scrollable card, right-align columns that are all numbers.
    for table in soup.find_all("table"):
        table.wrap(soup.new_tag("div", attrs={"class": "table-wrap"}))
        rows = [r.find_all(["td", "th"]) for r in table.select("tbody tr")]
        if not rows:
            continue
        for j in range(max(len(r) for r in rows)):
            col = [r[j] for r in rows if j < len(r)]
            if all(_NUMERIC.match(c.get_text(strip=True)) for c in col):
                for c in col:
                    c["class"] = c.get("class", []) + ["n"]
                header = table.select("thead tr")
                if header and j < len(header[-1].find_all("th")):
                    th = header[-1].find_all("th")[j]
                    th["class"] = th.get("class", []) + ["n"]

    # Headings -> flat list, then nested by level.
    # Numbering is automatic: the first NUMBER_DEPTH heading levels are
    # counted in order (1, 2, 3 / 1.1, 1.2 ...). Numbers typed by hand at
    # those levels are dropped, and "{-}" at the end opts a heading out.
    found = soup.find_all(["h1", "h2", "h3", "h4"])
    top_level = min((int(h.name[1]) for h in found), default=2)
    counters = [0] * (NUMBER_DEPTH + 1)
    flat = []
    for i, h in enumerate(found):
        if not h.get("id"):
            h["id"] = f"section-{i}"
        level = int(h.name[1])
        depth = level - top_level + 1
        h["class"] = h.get("class", []) + [f"lvl-{depth}"]
        texts = [
            s for s in h.find_all(string=True) if s.parent.name != "a" or "anchor-link" not in s.parent.get("class", [])
        ]

        unnumbered = False
        if texts and (m := _UNNUMBERED.search(str(texts[-1]))):
            texts[-1].replace_with(str(texts[-1])[: m.start()])
            unnumbered = True

        num = None
        if depth <= NUMBER_DEPTH:
            if texts and _NUM_PREFIX.match(str(texts[0])):  # hand-typed "1." / "1.2"
                texts[0].replace_with(_NUM_PREFIX.sub("", str(texts[0]), count=1))
        label = _heading_label(h)

        if depth <= NUMBER_DEPTH and not unnumbered:
            counters[depth] += 1
            counters[depth + 1 :] = [0] * (NUMBER_DEPTH - depth)
            num = ".".join(str(c) for c in counters[1 : depth + 1])
            tag = soup.new_tag("span", attrs={"class": "h-num" if depth == 1 else "h-subnum"})
            tag.string = f"Section {num}" if depth == 1 else num
            h.insert(0, tag)

        flat.append({"id": h["id"], "level": level, "num": num, "label": label, "children": []})

    tree: list[dict] = []
    stack: list[dict] = []
    for node in flat:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        (stack[-1]["children"] if stack else tree).append(node)
        stack.append(node)

    return str(soup), tree


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------

_ICON_SIDEBAR = (
    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="3" y="4" width="18" height="16" rx="2.5"/>'
    '<path d="M9 4v16"/></svg>'
)
_ICON_CHEVRON = (
    '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg>'
)
_ICON_UP = (
    '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
    'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 19V5"/><path d="m5 12 7-7 7 7"/></svg>'
)


def _render_toc(nodes: list[dict], depth: int = 1) -> str:
    items = []
    for n in nodes:
        num = f'<span class="num">{html.escape(n["num"])}</span>' if n["num"] else ""
        attrs = (
            f'href="#{html.escape(n["id"])}" data-target="{html.escape(n["id"])}" '
            f'data-label="{html.escape(n["label"])}" data-num="{html.escape(n["num"] or "")}"'
        )
        fold = ""
        children = ""
        if n["children"]:
            fold = f'<button class="toc-fold" type="button" aria-expanded="true" aria-label="Fold subsections">{_ICON_CHEVRON}</button>'
            children = _render_toc(n["children"], depth + 1)
        items.append(
            f'<li class="lvl-{depth}"><div class="toc-row"><a {attrs}>{num}<span class="label">{html.escape(n["label"])}</span></a>'
            f"{fold}</div>{children}</li>"
        )
    return f"<ul>{''.join(items)}</ul>"


_MATHJAX = """
<script>window.MathJax = { tex: { inlineMath: [['$', '$'], ['\\\\(', '\\\\)']], displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']], processEscapes: true } };</script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.2/es5/tex-chtml-full.js"></script>
"""

_HAS_MATH = re.compile(r"\$\$|\\\(|\\\[|\$[^\s$][^$\n]*\$")

_CSS = r"""
:root {
  --bg: #F2F5F6; --surface: #FFFFFF; --ink: #17232A; --muted: #566871; --line: #D3DCE0; --code-bg: #E9EEF0;
  --accent: #2350C4; --accent-soft: #E2E9FB; --fig-bg: #FFFFFF;
  --f-display: "Source Serif 4", Georgia, "Times New Roman", serif;
  --f-body: "Source Sans 3", "Segoe UI", system-ui, sans-serif;
  --f-mono: "IBM Plex Mono", ui-monospace, "SFMono-Regular", Menlo, monospace;
  --side-w: 290px; --bar-h: 52px;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #10171B; --surface: #172127; --ink: #E4ECEF; --muted: #98AAB3; --line: #2A3940; --code-bg: #1D2A30;
    --accent: #86A6FF; --accent-soft: #1E2B4C; color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #10171B; --surface: #172127; --ink: #E4ECEF; --muted: #98AAB3; --line: #2A3940; --code-bg: #1D2A30;
  --accent: #86A6FF; --accent-soft: #1E2B4C; color-scheme: dark;
}
* { box-sizing: border-box; }
html { scroll-padding-top: calc(var(--bar-h) + 20px); }
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior: smooth; } }
body { margin: 0; background: var(--bg); color: var(--ink); font-family: var(--f-body); font-size: 16.5px; line-height: 1.6; -webkit-font-smoothing: antialiased; }
a { color: var(--accent); }
button { font: inherit; }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ---------- layout ---------- */
.layout { min-height: 100vh; }
.sidebar { background: var(--surface); border-right: 1px solid var(--line); z-index: 30; overflow: hidden; }
.sidebar-inner { width: var(--side-w); height: 100%; display: flex; flex-direction: column; }
.main { min-width: 0; }

@media (min-width: 961px) {
  .layout { display: grid; grid-template-columns: var(--side-w) minmax(0, 1fr); transition: grid-template-columns .22s ease; }
  .sidebar { position: sticky; top: 0; height: 100vh; transition: visibility 0s linear 0s; }
  html.toc-closed .layout { grid-template-columns: 0px minmax(0, 1fr); }
  html.toc-closed .sidebar { visibility: hidden; border-right-color: transparent; transition: visibility 0s linear .22s; }
  .scrim { display: none; }
}
@media (max-width: 960px) {
  .sidebar { position: fixed; inset: 0 auto 0 0; width: min(86vw, var(--side-w)); transform: translateX(-100%); visibility: hidden;
             transition: transform .22s ease, visibility 0s linear .22s; box-shadow: 0 10px 40px rgb(0 0 0 / .18); }
  .sidebar-inner { width: 100%; }
  html.toc-open .sidebar { transform: none; visibility: visible; transition: transform .22s ease; }
  .scrim { position: fixed; inset: 0; z-index: 25; background: rgb(8 12 14 / .4); opacity: 0; pointer-events: none; transition: opacity .2s; }
  html.toc-open .scrim { opacity: 1; pointer-events: auto; }
}

/* ---------- sidebar ---------- */
.side-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 10px; padding: 18px 14px 14px 20px; border-bottom: 1px solid var(--line); }
.side-head .eyebrow { margin-bottom: 4px; }
.side-title { font-family: var(--f-display); font-weight: 700; font-size: 15px; line-height: 1.25; text-wrap: balance; }
.toc { flex: 1; overflow-y: auto; padding: 12px 10px 28px; overscroll-behavior: contain; }
.toc ul { list-style: none; margin: 0; padding: 0; }
.toc ul ul { margin: 2px 0 8px 17px; padding-left: 8px; border-left: 1px solid var(--line); }
.toc li.lvl-1 + li.lvl-1 { margin-top: 4px; }
.toc-row { display: flex; align-items: flex-start; gap: 2px; }
.toc a { flex: 1; min-width: 0; display: flex; gap: 8px; padding: 6px 10px; border-radius: 7px; color: var(--muted); text-decoration: none; font-size: 14px; line-height: 1.35; }
.toc a:hover { background: var(--code-bg); color: var(--ink); }
.toc li.lvl-1 > .toc-row > a { color: var(--ink); font-weight: 700; }
.toc a .num { font-family: var(--f-mono); font-size: 12px; font-weight: 600; color: var(--accent); min-width: 1.2em; padding-top: 1px; }
.toc li > .toc-row > a.in-path { color: var(--accent); }
.toc li > .toc-row > a.active { background: var(--accent-soft); color: var(--accent); }
.toc-fold { flex: none; width: 28px; height: 30px; display: grid; place-items: center; border: 0; background: none; color: var(--muted); border-radius: 6px; cursor: pointer; }
.toc-fold:hover { background: var(--code-bg); color: var(--ink); }
.toc-fold svg { transition: transform .15s; }
.toc li.folded > ul { display: none; }
.toc li.folded > .toc-row .toc-fold svg { transform: rotate(-90deg); }

/* ---------- top bar ---------- */
.topbar { position: sticky; top: 0; z-index: 20; height: var(--bar-h); display: flex; align-items: center; gap: 12px; padding: 0 16px;
          background: color-mix(in srgb, var(--bg) 86%, transparent); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); border-bottom: 1px solid var(--line); }
.icon-btn { flex: none; width: 34px; height: 34px; display: grid; place-items: center; border: 1px solid var(--line); background: var(--surface); color: var(--ink); border-radius: 8px; cursor: pointer; text-decoration: none; }
.icon-btn:hover { border-color: var(--muted); }
.crumb { flex: 1; min-width: 0; font-size: 14px; color: var(--muted); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.crumb .sep { margin: 0 6px; opacity: .6; }
.crumb b { color: var(--ink); font-weight: 700; }
.crumb .num { font-family: var(--f-mono); font-size: 12px; color: var(--accent); margin-right: 6px; }
.progress { position: absolute; left: 0; right: 0; bottom: -1px; height: 2px; background: var(--accent); transform-origin: left; transform: scaleX(0); }

/* ---------- content ---------- */
.content { max-width: 920px; margin: 0 auto; padding: 0 28px 96px; }
.hero { padding: 56px 0 8px; display: grid; gap: 14px; }
.eyebrow { font-family: var(--f-mono); font-size: 11.5px; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
.hero h1 { font-family: var(--f-display); font-size: clamp(32px, 5vw, 48px); font-weight: 700; line-height: 1.08; letter-spacing: -.02em; margin: 0; text-wrap: balance; }
.lede { margin: 0; max-width: 66ch; font-size: 18px; line-height: 1.5; color: var(--muted); }
.meta { display: flex; flex-wrap: wrap; gap: 6px 8px; margin-top: 4px; }
.meta a { color: inherit; }
.meta code { font-family: inherit; }
.meta span { font-family: var(--f-mono); font-size: 12px; color: var(--muted); background: var(--surface); border: 1px solid var(--line); border-radius: 999px; padding: 3px 10px; }

.report h1, .report h2, .report h3, .report h4 { font-family: var(--f-display); line-height: 1.2; margin: 0; text-wrap: balance; }
.report h2 { font-size: clamp(25px, 3.4vw, 31px); letter-spacing: -.015em; margin: 64px 0 18px; padding-top: 28px; border-top: 1px solid var(--line); }
.report h2 .h-num { display: block; margin-bottom: 8px; font-family: var(--f-mono); font-size: 12px; font-weight: 600; letter-spacing: .08em; text-transform: uppercase; color: var(--accent); }
.report .h-subnum { margin-right: .5em; font-family: var(--f-mono); font-size: .78em; font-weight: 600; color: var(--accent); }
.report h3 { font-size: 21px; margin: 44px 0 12px; }
.report h4 { font-size: 17.5px; margin: 34px 0 10px; }
.report .anchor-link { margin-left: .35em; color: var(--muted); text-decoration: none; font-weight: 400; font-size: .75em; opacity: 0; transition: opacity .15s; }
.report :is(h1, h2, h3, h4):hover .anchor-link { opacity: 1; }
.report p, .report ul, .report ol { margin: 0 0 14px; }
.report li + li { margin-top: 4px; }
.report strong { font-weight: 700; }
.report a { overflow-wrap: anywhere; }
.report hr { border: 0; margin: 24px 0; }
.report blockquote { margin: 18px 0; padding: 10px 16px; border-left: 3px solid var(--accent); background: var(--surface); border-radius: 0 8px 8px 0; color: var(--muted); }
.report code { font-family: var(--f-mono); font-size: .86em; background: var(--code-bg); padding: 1px 5px; border-radius: 4px; }
.report pre { font-family: var(--f-mono); font-size: 13px; line-height: 1.5; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; padding: 12px 14px; overflow-x: auto; }
.report pre code { background: none; padding: 0; }

.fig { margin: 22px 0 28px; padding: 10px; display: grid; gap: 10px; background: var(--fig-bg); border: 1px solid var(--line); border-radius: 12px; }
.fig img, .output_png img, .output_jpeg img { display: block; max-width: 100%; height: auto; margin: 0 auto; cursor: zoom-in; }
.output_png, .output_jpeg, .output_svg { margin: 22px 0 28px; padding: 10px; background: var(--fig-bg); border: 1px solid var(--line); border-radius: 12px; }

.table-wrap { overflow-x: auto; margin: 18px 0 26px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
.report table { border-collapse: collapse; width: 100%; font-size: 14.5px; font-variant-numeric: tabular-nums; }
.report th, .report td { padding: 9px 14px; text-align: left; border-bottom: 1px solid var(--line); white-space: nowrap; }
.report th { font-size: 12.5px; font-weight: 700; color: var(--muted); background: var(--code-bg); }
.report tbody tr:last-child td { border-bottom: 0; }
.report tbody tr:hover td { background: color-mix(in srgb, var(--accent-soft) 45%, transparent); }
.report .n { text-align: right; }

.foot { margin-top: 80px; padding-top: 18px; border-top: 1px solid var(--line); font-family: var(--f-mono); font-size: 12px; color: var(--muted); }

/* ---------- image zoom ---------- */
.lightbox { position: fixed; inset: 0; z-index: 60; display: none; overflow: auto; background: rgb(8 12 14 / .9); cursor: zoom-out; }
.lightbox.open { display: flex; }
.lightbox img { margin: auto; max-width: 96vw; max-height: 94vh; background: #fff; border-radius: 6px; cursor: zoom-in; }
.lightbox.actual img { max-width: none; max-height: none; cursor: zoom-out; }

@media (max-width: 640px) {
  body { font-size: 16px; }
  .content { padding: 0 16px 72px; }
  .hero { padding-top: 36px; }
  .topbar { padding: 0 12px; }
  .crumb .parent { display: none; }
  .fig { padding: 6px; }
}
@media print {
  .sidebar, .topbar, .scrim, .lightbox { display: none !important; }
  .layout { display: block; }
  .report h2 { break-before: page; }
}
"""

_JS = r"""
(function () {
  var root = document.documentElement;
  var mobile = window.matchMedia('(max-width: 960px)');
  var KEY = 'report-toc-closed';
  var toggles = document.querySelectorAll('[data-toc-toggle]');

  function isOpen() { return mobile.matches ? root.classList.contains('toc-open') : !root.classList.contains('toc-closed'); }
  function sync() { var o = isOpen(); toggles.forEach(function (b) { b.setAttribute('aria-expanded', String(o)); }); }
  function setOpen(open) {
    if (mobile.matches) { root.classList.toggle('toc-open', open); }
    else {
      root.classList.toggle('toc-closed', !open);
      try { localStorage.setItem(KEY, open ? '0' : '1'); } catch (e) {}
    }
    sync();
  }
  toggles.forEach(function (b) { b.addEventListener('click', function () { setOpen(!isOpen()); }); });
  document.querySelector('.scrim').addEventListener('click', function () { setOpen(false); });
  mobile.addEventListener('change', function () { root.classList.remove('toc-open'); sync(); });
  sync();

  // fold / unfold subsections
  document.querySelectorAll('.toc-fold').forEach(function (b) {
    b.addEventListener('click', function () {
      var li = b.closest('li');
      var folded = li.classList.toggle('folded');
      b.setAttribute('aria-expanded', String(!folded));
      b.setAttribute('aria-label', folded ? 'Unfold subsections' : 'Fold subsections');
    });
  });

  // scroll spy + breadcrumb + progress
  var toc = document.querySelector('.toc');
  var crumb = document.querySelector('.crumb');
  var progress = document.querySelector('.progress');
  var defaultCrumb = crumb.textContent;
  var links = Array.prototype.slice.call(document.querySelectorAll('.toc a[data-target]'));
  var items = links.map(function (a) { return { a: a, el: document.getElementById(a.dataset.target) }; })
                   .filter(function (it) { return it.el; });
  var current;

  links.forEach(function (a) { a.addEventListener('click', function () { if (mobile.matches) setOpen(false); }); });

  function renderCrumb(path) {
    crumb.textContent = '';
    if (!path.length) { crumb.textContent = defaultCrumb; return; }
    var last = path.length - 1;
    path.forEach(function (a, i) {
      if (i) { var s = document.createElement('span'); s.className = 'sep parent'; s.textContent = '›'; crumb.appendChild(s); }
      if (i === 0 && a.dataset.num) { var n = document.createElement('span'); n.className = 'num'; n.textContent = a.dataset.num; crumb.appendChild(n); }
      var t = document.createElement(i === last ? 'b' : 'span');
      if (i !== last) t.className = 'parent';
      t.textContent = a.dataset.label;
      crumb.appendChild(t);
    });
  }

  function update() {
    var doc = document.documentElement;
    var max = doc.scrollHeight - doc.clientHeight;
    var p = max > 0 ? Math.min(1, doc.scrollTop / max) : 0;
    progress.style.transform = 'scaleX(' + p + ')';

    var line = (parseFloat(getComputedStyle(root).getPropertyValue('--bar-h')) || 52) + 32;
    var act = null;
    for (var i = 0; i < items.length; i++) {
      if (items[i].el.getBoundingClientRect().top - line <= 0) act = items[i]; else break;
    }
    if (p > 0.995 && items.length) act = items[items.length - 1];
    if (act === current) return;
    current = act;

    links.forEach(function (a) { a.classList.remove('active', 'in-path'); a.removeAttribute('aria-current'); });
    if (!act) { renderCrumb([]); return; }
    act.a.classList.add('active');
    act.a.setAttribute('aria-current', 'location');
    var path = [act.a];
    var li = act.a.closest('li').parentElement.closest('li');
    while (li) {
      var pa = li.querySelector(':scope > .toc-row > a');
      pa.classList.add('in-path');
      path.unshift(pa);
      li = li.parentElement.closest('li');
    }
    renderCrumb(path);

    var r = act.a.getBoundingClientRect(), tr = toc.getBoundingClientRect();
    if (r.top < tr.top + 8 || r.bottom > tr.bottom - 8) toc.scrollTop += r.top - tr.top - tr.height / 3;
  }
  var ticking = false;
  window.addEventListener('scroll', function () {
    if (!ticking) { ticking = true; requestAnimationFrame(function () { update(); ticking = false; }); }
  }, { passive: true });
  window.addEventListener('resize', update);
  update();

  // click a figure to view it larger; click again for actual size
  var box = document.querySelector('.lightbox');
  var boxImg = box.querySelector('img');
  document.querySelectorAll('.fig img, .output_png img, .output_jpeg img').forEach(function (img) {
    img.addEventListener('click', function () {
      boxImg.src = img.src; boxImg.alt = img.alt || '';
      box.classList.remove('actual'); box.classList.add('open');
    });
  });
  boxImg.addEventListener('click', function (e) { e.stopPropagation(); box.classList.toggle('actual'); });
  box.addEventListener('click', function () { box.classList.remove('open'); });
  document.addEventListener('keydown', function (e) {
    if (e.key !== 'Escape') return;
    if (box.classList.contains('open')) box.classList.remove('open');
    else if (mobile.matches && isOpen()) setOpen(false);
  });
})();
"""

_PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>%%TITLE%%</title>
<script>try { if (localStorage.getItem('report-toc-closed') === '1') document.documentElement.classList.add('toc-closed'); } catch (e) {}</script>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&family=Source+Sans+3:ital,wght@0,400;0,600;0,700;1,400&family=IBM+Plex+Mono:wght@400;600&display=swap">
<style>%%CSS%%</style>
%%MATHJAX%%
</head>
<body>
<div class="layout">
  <aside class="sidebar" id="sidebar" aria-label="Contents">
    <div class="sidebar-inner">
      <div class="side-head">
        <div>
          <div class="eyebrow">Contents</div>
          <div class="side-title">%%TITLE%%</div>
        </div>
        <button class="icon-btn" type="button" data-toc-toggle aria-controls="sidebar" aria-label="Hide contents" title="Hide contents">%%ICON_SIDEBAR%%</button>
      </div>
      <nav class="toc">%%TOC%%</nav>
    </div>
  </aside>
  <div class="scrim"></div>
  <div class="main">
    <header class="topbar">
      <button class="icon-btn" type="button" data-toc-toggle aria-controls="sidebar" aria-label="Show or hide contents" title="Contents">%%ICON_SIDEBAR%%</button>
      <div class="crumb" aria-live="polite">%%TITLE%%</div>
      <a class="icon-btn" href="#top" aria-label="Back to top" title="Back to top">%%ICON_UP%%</a>
      <div class="progress"></div>
    </header>
    <main class="content" id="top">
      <header class="hero">
        %%EYEBROW%%
        <h1>%%TITLE%%</h1>
        %%LEDE%%
        <div class="meta">%%META%%</div>
      </header>
      <article class="report">
%%BODY%%
      </article>
      <footer class="foot">Built from %%SOURCE%% on %%DATE%%</footer>
    </main>
  </div>
</div>
<div class="lightbox" role="dialog" aria-label="Figure, enlarged"><img alt=""></div>
<script>%%JS%%</script>
</body>
</html>
"""


def render_page(header: dict, body: str, tree: list[dict], source_name: str, has_math: bool) -> str:
    today = dt.date.today()
    date_str = f"{today.day} {today:%b %Y}"
    title = header["title"]
    pills = header["pills"] + [html.escape(f"Updated {date_str}")]
    replacements = {
        "%%CSS%%": _CSS,
        "%%JS%%": _JS,
        "%%MATHJAX%%": _MATHJAX if has_math else "",
        "%%TOC%%": _render_toc(tree),
        "%%ICON_SIDEBAR%%": _ICON_SIDEBAR,
        "%%ICON_UP%%": _ICON_UP,
        "%%EYEBROW%%": f'<div class="eyebrow">{header["eyebrow"]}</div>' if header["eyebrow"] else "",
        "%%LEDE%%": f'<p class="lede">{header["lede"]}</p>' if header["lede"] else "",
        "%%META%%": "".join(f"<span>{p}</span>" for p in pills),
        "%%SOURCE%%": html.escape(source_name),
        "%%DATE%%": date_str,
        "%%TITLE%%": html.escape(title),
    }
    page = _PAGE
    for key, value in replacements.items():
        if key != "%%BODY%%":
            page = page.replace(key, value)
    # Body last, so nothing inside the report is mistaken for a placeholder.
    return page.replace("%%BODY%%", body)


def build_html(notebook_path: Path, output_path: Path) -> None:
    nb = nbformat.read(notebook_path, as_version=4)
    embed_local_images(nb, notebook_path.parent)
    header = pop_header(nb, fallback_title=notebook_path.stem)
    has_math = any(c.cell_type == "markdown" and _HAS_MATH.search(c.source) for c in nb.cells)

    exporter = HTMLExporter(
        template_name="basic", exclude_input=True, exclude_input_prompt=True, exclude_output_prompt=True
    )
    body, _ = exporter.from_notebook_node(nb)
    body, tree = enhance_body(body)

    page = render_page(header, body, tree, notebook_path.name, has_math)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")


def main():
    notebook_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_NOTEBOOK
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_OUTPUT

    build_html(notebook_path, output_path)
    print(f"wrote {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
