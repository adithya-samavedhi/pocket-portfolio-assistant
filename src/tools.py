"""The retrieval tool boundary (Milestone 6).

`search_filings` is the single entry point both the answering agent and the MCP
server call. It returns structured summaries (metadata + passage text), never a
raw dump — the discipline the plan locks in Phase 1 so it's habitual by Phase 2.
"""
import re
from typing import List, Dict, Optional

from retrieval import Retriever

_retriever: Optional[Retriever] = None


def _get_retriever() -> Retriever:
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def _format(hit: Dict) -> Dict:
    m = hit["metadata"]
    return {
        "ticker": m["ticker"],
        "filing_type": m["filing_type"],
        "fiscal_period": m["fiscal_period"],
        # Self-describing: a human reading "Q2 FY2026 (ended 2025-07-27)" cannot
        # mistake it for calendar Q2, which is what the old labels invited.
        "period_label": f"{m['fiscal_period']} (ended {m.get('period_end', '?')})",
        "period_end": m.get("period_end", ""),
        "period_type": m.get("period_type", ""),
        "calendar_quarter": m.get("calendar_quarter", ""),
        "section": m["section"],
        "chunk_kind": m["chunk_kind"],
        "source_url": m["source_url"],
        "text": hit["document"],
    }


def _period_clause(period: str):
    """Match a period the user typed against either labelling scheme.

    "Q2 FY2026" and "FY2025" are unambiguous. A bare "Q2 2026" is not — it may
    mean the company's fiscal Q2 or the calendar quarter, which for an
    off-calendar filer are different documents. Rather than guess or interrogate
    the user, match both and let the answer state which period it used.
    """
    p = period.strip()
    alts = [{"fiscal_period": p}, {"calendar_quarter": p}]
    m = re.fullmatch(r"(Q[1-4])\s+(\d{4})", p, re.I)
    if m:                                   # "Q2 2026" -> also try "Q2 FY2026"
        alts.append({"fiscal_period": f"{m.group(1).upper()} FY{m.group(2)}"})
    return alts[0] if len(alts) == 1 else {"$or": alts}


def _where(ticker, period, filing_type, period_type=None):
    clauses = []
    if ticker:
        clauses.append({"ticker": ticker})
    if period:
        clauses.append(_period_clause(period))
    if filing_type:
        clauses.append({"filing_type": filing_type})
    if period_type:
        clauses.append({"period_type": period_type})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def search_filings(query: str, ticker: str = None, section: str = None,
                   period: str = None, filing_type: str = None,
                   k: int = 6, temporal: bool = False,
                   period_type: str = None) -> List[Dict]:
    """Retrieve filing passages.

    query       natural-language question or keywords
    ticker      restrict to one company (e.g. "NVDA")
    section     substring of the SEC item label (e.g. "Item 1A")
    period      the company's own label ("Q2 FY2026", "FY2025"); a bare
                "Q2 2026" matches both the fiscal and calendar reading
    filing_type "10-K" or "10-Q"
    period_type "quarter" or "annual"
    k           number of passages (per period when temporal=True)
    temporal    fan out across a ticker's quarters so each one is represented
                (requires ticker) — for change-over-time questions. Annual
                filings are excluded: a fiscal year contains its own quarters.
    """
    r = _get_retriever()

    if temporal and ticker:
        groups = r.temporal_search(query, ticker, k_per_period=k,
                                   filing_type=filing_type, section=section,
                                   period_type=period_type or "quarter")
        return [_format(h) for g in groups for h in g["hits"]]

    where = _where(ticker, period, filing_type, period_type)
    pool = k * 6 if section else k
    hits = r.search(query, k=pool, where=where)
    if section:
        needle = section.lower()
        hits = [h for h in hits if needle in h["metadata"].get("section", "").lower()]
    return [_format(h) for h in hits[:k]]
