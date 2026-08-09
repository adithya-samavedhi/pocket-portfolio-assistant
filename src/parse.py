"""Parse SEC filing HTML into a narrative stream plus extracted tables.

Milestone 2 keeps financial tables whole instead of shredding them across
character chunks. Real data tables are pulled out and rendered as pipe-
delimited markdown; layout/spacer tables are flattened back into narrative.
The narrative text carries a placeholder token (``\\x00TBLn\\x00``) at each
extracted table's original position so section ordering is preserved.
"""
import re
import warnings

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning

# SEC filings are inline XBRL (XML-ish); the HTML parser handles them fine, so
# quiet the "this looks like XML" advisory rather than switching parsers.
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

TABLE_TOKEN = re.compile(r"\x00TBL(\d+)\x00")

# Block-level tags that should force a line break during text extraction.
_BLOCK_TAGS = ["p", "div", "br", "li", "tr", "section", "hr",
               "h1", "h2", "h3", "h4", "h5", "h6"]


def _normalize_ws(text: str) -> str:
    text = text.replace("\xa0", " ")  # non-breaking spaces are rife in filings
    text = re.sub(r"[\t\r\f\v]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def _is_data_table(tbl) -> bool:
    """A real financial table: multiple rows and several digit-bearing cells.

    SEC HTML wraps a lot of layout in <table>; those are mostly empty spacers.
    """
    cells = tbl.find_all(["td", "th"])
    numeric_cells = sum(1 for c in cells if re.search(r"\d", c.get_text()))
    return len(tbl.find_all("tr")) >= 2 and numeric_cells >= 4


def table_to_markdown(tbl) -> str:
    """Render a table tag as pipe-delimited rows, dropping empty spacer cells."""
    lines = []
    for tr in tbl.find_all("tr"):
        cells = [re.sub(r"\s+", " ", c.get_text(" ", strip=True))
                 for c in tr.find_all(["td", "th"])]
        cells = [c for c in cells if c]  # spacer cells are empty; drop them
        if cells:
            lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def html_to_stream(html: str):
    """Return (narrative_text_with_placeholders, [table_markdown, ...]).

    Data tables are replaced by ``\\x00TBLn\\x00`` tokens in the narrative;
    layout tables are flattened into surrounding text.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "head", "title"]):
        tag.decompose()

    tables = []
    for tbl in soup.find_all("table"):
        if _is_data_table(tbl):
            tables.append(table_to_markdown(tbl))
            tbl.replace_with(soup.new_string(f" \x00TBL{len(tables) - 1}\x00 "))
        else:
            tbl.replace_with(soup.new_string(" " + tbl.get_text(" ", strip=True) + " "))

    # Some filers (e.g. MSFT) split words across inline <span>s. Joining with a
    # space separator would turn "RISK" into "RIS K"; instead mark block-level
    # boundaries with newlines and join everything else tight.
    for tag in soup.find_all(_BLOCK_TAGS):
        tag.append("\n")

    return _normalize_ws(soup.get_text("")), tables


def load_and_parse(path):
    """Read a raw filing and return (narrative_text, tables)."""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return html_to_stream(f.read())
