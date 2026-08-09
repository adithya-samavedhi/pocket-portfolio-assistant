"""Fundamentals sub-agent (Phase 2, Milestone 1).

The Phase 1 filings RAG, promoted into a callable specialist that returns the
shared `AgentAnswer` contract. It retrieves grounded passages internally, has an
LLM answer strictly from them, and hands back a compact summary — answer plus
the sources it actually cited, confidence, and an insufficient-evidence flag.

The verbose retrieved passages never leave this object: only citation metadata
does. That is what keeps the orchestrator's context bounded.
"""
from typing import Optional

from contracts import AgentAnswer, Citation, Confidence, renumber
from llm import LLM
from tools import search_filings

SYSTEM = """You are the Fundamentals specialist in a multi-agent finance system. \
Answer the question using ONLY the numbered source passages from SEC filings. \
Each source is labeled with company, filing type, fiscal period, and section. \
Sources marked [TABLE] are financial tables — that is where the figures live, \
so read them before concluding a number is unavailable.

Return a single JSON object with exactly these keys:
  "answer": a specific, substantive answer (<= 200 words), citing sources inline
            as [1], [2].
  "cited": array of the source numbers you actually used, e.g. [1, 3].
  "confidence": one of "low", "medium", "high".
  "insufficient_evidence": true ONLY if the sources genuinely cannot answer it.

How to answer well:
- LEAD with the direct answer, then support it with figures. Name the specific
  segments, products, risks or drivers — never "several factors".
- QUOTE the actual numbers from the sources, with their period and units
  ("Compute & Networking revenue was $41,096M in Q2 FY2026"). A figure without
  a period is useless.
- For change-over-time questions, give the value for EACH period you have, in
  chronological order, then state the direction and size of the change.
- You may compute a difference or a percent change, but ONLY from figures that
  appear in the sources, and you must show the underlying figures you used.
  Never estimate, extrapolate, or carry in outside knowledge.
- If the question's premise is wrong, say so and answer what the filings DO
  support (e.g. if it asks for the top 3 segments but the company reports two,
  state that it reports two and give both).
- If the sources answer part of the question, answer that part and name exactly
  what is missing. Reserve insufficient_evidence=true for when they support
  nothing at all — a missing preferred format is not missing evidence.

When you do set insufficient_evidence=true, use cited=[] and say plainly what \
was not in the sources. Output JSON only, no prose around it."""

_CITE_FIELDS = ("ticker", "filing_type", "fiscal_period", "section", "source_url")


class FundamentalsAgent:
    name = "fundamentals"

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()

    def _context(self, sources) -> str:
        blocks = []
        for i, s in enumerate(sources, 1):
            label = f"{s['ticker']} {s['filing_type']} {s['fiscal_period']} — {s['section']}"
            # Flag tables explicitly: the model otherwise reads a pipe-delimited
            # grid as prose and misses that it is looking at the figures.
            if s.get("chunk_kind") == "table":
                label += " [TABLE]"
            blocks.append(f"[{i}] ({label})\n{' '.join(s['text'].split())}")
        return "\n\n".join(blocks)

    def answer(self, question: str, ticker: str = None, section: str = None,
               period: str = None, filing_type: str = None,
               k: int = 6, temporal: bool = False) -> AgentAnswer:
        sources = search_filings(question, ticker=ticker, section=section,
                                 period=period, filing_type=filing_type,
                                 k=k, temporal=temporal)
        if not sources:
            return AgentAnswer(agent=self.name, insufficient_evidence=True,
                               answer="No filing passages matched the query/filters.")

        data = self.llm.complete_json(
            SYSTEM, f"Sources:\n{self._context(sources)}\n\nQuestion: {question}")

        # Map the numbers the model cited back to source metadata (no passage
        # text). Distinct chunks can share a section, so dedupe to a clean list
        # and remember where each original number landed, so the inline markers
        # can be renumbered to match rather than pointing past the list.
        citations, seen, remap = [], {}, {}
        for n in data.get("cited", []) or []:
            if isinstance(n, int) and 1 <= n <= len(sources):
                s = sources[n - 1]
                key = tuple(s[f] for f in _CITE_FIELDS)
                if key not in seen:
                    citations.append(Citation(**dict(zip(_CITE_FIELDS, key))))
                    seen[key] = len(citations)
                remap[n] = seen[key]

        confidence: Confidence = data.get("confidence", "low")
        if confidence not in ("low", "medium", "high"):
            confidence = "low"

        return AgentAnswer(
            agent=self.name,
            answer=renumber(str(data.get("answer", "")).strip(), remap),
            citations=citations,
            confidence=confidence,
            insufficient_evidence=bool(data.get("insufficient_evidence", False)),
        )
