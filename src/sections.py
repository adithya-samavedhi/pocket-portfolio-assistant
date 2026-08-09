"""Detect 10-K / 10-Q Item boundaries and chunk within them (Milestone 2).

Filers format headings inconsistently: some UPPERCASE them, some use title case
identical to their cross-references ("see Item 1A. Risk Factors ..."). So we
don't key on case. Instead we match canonical SEC item titles in document
order, and reject cross-references — which are quoted, or preceded by cues like
"Refer to" / "in conjunction with", or followed by "of this Form" / a comma.

Chunks never cross a section boundary; each carries its section label.
"""
import re
from typing import List, Dict

from parse import TABLE_TOKEN
from chunk import fixed_size_chunks
from config import CHUNK_SIZE, MIN_CHUNK_CHARS


def section_code(label: str) -> str:
    """'Item 1A: Risk Factors' -> 'Item 1A' (so 'Item 1' won't match 'Item 1A')."""
    return label.split(":")[0].strip()

# Canonical items in document order: (number, title-regex, clean label).
CANON_10K = [
    ("1",  r"Business", "Business"),
    ("1A", r"Risk\s+Factors", "Risk Factors"),
    ("1B", r"Unresolved\s+Staff\s+Comments", "Unresolved Staff Comments"),
    ("1C", r"Cybersecurity", "Cybersecurity"),
    ("2",  r"Properties", "Properties"),
    ("3",  r"Legal\s+Proceedings", "Legal Proceedings"),
    ("4",  r"Mine\s+Safety\s+Disclosures", "Mine Safety Disclosures"),
    ("5",  r"Market\s+for\s+(?:the\s+)?Registrant[’'`]?s?", "Market for Registrant's Common Equity"),
    ("7",  r"Management[’'`]s\s+Discussion\s+and\s+Analysis"
           r"(?:\s+of\s+Financial\s+Condition\s+and\s+Results\s+of\s+Operations)?",
           "Management's Discussion and Analysis"),
    ("7A", r"Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk",
           "Quantitative and Qualitative Disclosures"),
    ("8",  r"Financial\s+Statements\s+and\s+Supplementary\s+Data", "Financial Statements"),
    ("9",  r"Changes\s+in\s+and\s+Disagreements\s+[Ww]ith\s+Accountants",
           "Changes in and Disagreements with Accountants"),
    ("9A", r"Controls\s+and\s+Procedures", "Controls and Procedures"),
    ("9B", r"Other\s+Information", "Other Information"),
    ("10", r"Directors,?\s+Executive\s+Officers", "Directors and Corporate Governance"),
    ("11", r"Executive\s+Compensation", "Executive Compensation"),
    ("12", r"Security\s+Ownership", "Security Ownership"),
    ("13", r"Certain\s+Relationships", "Certain Relationships"),
    ("14", r"Principal\s+Accountant", "Principal Accountant Fees and Services"),
    ("15", r"Exhibit", "Exhibits"),
]

# 10-Q items span Part I then Part II; item numbers repeat but titles differ, so
# the in-order match with monotonic positions disambiguates them.
CANON_10Q = [
    ("1",  r"Financial\s+Statements", "Financial Statements"),
    ("2",  r"Management[’'`]s\s+Discussion\s+and\s+Analysis"
           r"(?:\s+of\s+Financial\s+Condition\s+and\s+Results\s+of\s+Operations)?",
           "Management's Discussion and Analysis"),
    ("3",  r"Quantitative\s+and\s+Qualitative\s+Disclosures\s+About\s+Market\s+Risk",
           "Quantitative and Qualitative Disclosures"),
    ("4",  r"Controls\s+and\s+Procedures", "Controls and Procedures"),
    ("1",  r"Legal\s+Proceedings", "Legal Proceedings"),
    ("1A", r"Risk\s+Factors", "Risk Factors"),
    ("2",  r"Unregistered\s+Sales\s+of\s+Equity\s+Securities", "Unregistered Sales of Equity Securities"),
    ("3",  r"Defaults\s+Upon\s+Senior\s+Securities", "Defaults Upon Senior Securities"),
    ("5",  r"Other\s+Information", "Other Information"),
    ("6",  r"Exhibit", "Exhibits"),
]

_QUOTES = "\"'“”‘’"
_PRE_CUE = re.compile(
    r"(?:refer to|conjunction with|see|under|within|described in|set forth in|"
    r"discussed in|part\s+i{1,3}\s*,)\s*$", re.I)
_POST_CUE = re.compile(
    r"^\s*(?:[" + _QUOTES + r"]|,|;|for\b|in\s+(?:our|this)|of\s+(?:this|our)|"
    r"as\b|above|below)", re.I)

# A table rendered as its own chunk is capped near the embedder's window; larger
# tables are split on row boundaries with the header row repeated.
_TABLE_CHUNK = 1500


def _is_crossref(text: str, start: int, title_end: int) -> bool:
    pre = text[:start].rstrip()
    if pre and pre[-1] in _QUOTES:            # "...Item 1A. Risk Factors" (quoted)
        return True
    if _PRE_CUE.search(pre[-30:]):            # "Refer to Item 1A...", "see Item 7"
        return True
    if _POST_CUE.match(text[title_end:title_end + 20]):  # "...Data in our Annual Report"
        return True
    return False


def detect_sections(text: str, form: str = "10-K") -> List[Dict]:
    """Return ordered [{label, start, end}] spans covering the whole text."""
    canon = CANON_10Q if form == "10-Q" else CANON_10K
    marks, pos = [], -1
    for num, title_rx, label in canon:
        rx = re.compile(r"Item\s+" + re.escape(num) + r"(?![A-Za-z0-9])[.\s]+(" + title_rx + r")", re.I)
        for m in rx.finditer(text):
            if m.start() <= pos or _is_crossref(text, m.start(), m.end()):
                continue
            marks.append((m.start(), f"Item {num}: {label}"))
            pos = m.start()
            break

    if not marks:
        return [{"label": "Full Document", "start": 0, "end": len(text)}]

    spans = []
    if marks[0][0] > 0:
        spans.append({"label": "Front Matter", "start": 0, "end": marks[0][0]})
    for i, (start, label) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        spans.append({"label": label, "start": start, "end": end})
    return spans


def _split_table(md: str) -> List[str]:
    if len(md) <= _TABLE_CHUNK:
        return [md]
    rows = md.split("\n")
    header = rows[0]
    out, buf = [], [header]
    for row in rows[1:]:
        buf.append(row)
        if sum(len(r) for r in buf) >= _TABLE_CHUNK:
            out.append("\n".join(buf))
            buf = [header]
    if len(buf) > 1:
        out.append("\n".join(buf))
    return out


def chunk_filing(text: str, tables: List[str], form: str = "10-K") -> List[Dict]:
    """Section-aware chunks. Each: {text, section, kind ('text'|'table')}."""
    chunks: List[Dict] = []
    for span in detect_sections(text, form):
        section, segment = span["label"], text[span["start"]:span["end"]]
        pos = 0
        for m in TABLE_TOKEN.finditer(segment):
            for piece in fixed_size_chunks(segment[pos:m.start()], CHUNK_SIZE):
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append({"text": piece, "section": section, "kind": "text"})
            for piece in _split_table(tables[int(m.group(1))]):
                chunks.append({"text": piece, "section": section, "kind": "table"})
            pos = m.end()
        for piece in fixed_size_chunks(segment[pos:], CHUNK_SIZE):
            if len(piece) > 40:
                chunks.append({"text": piece, "section": section, "kind": "text"})
    return chunks
