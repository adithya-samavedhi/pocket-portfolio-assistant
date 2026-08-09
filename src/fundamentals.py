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
Each source is labeled with company, filing type, fiscal period, and section.

Return a single JSON object with exactly these keys:
  "answer": a concise answer (<= 120 words), citing sources inline as [1], [2].
            Use only facts in the sources; never invent or estimate figures.
            For change-over-time questions, compare the periods explicitly.
  "cited": array of the source numbers you actually used, e.g. [1, 3].
  "confidence": one of "low", "medium", "high".
  "insufficient_evidence": true if the sources do not answer the question.

If the sources don't answer it, set insufficient_evidence=true, cited=[], and \
say so plainly in "answer" — do not guess. Output JSON only, no prose around it."""

_CITE_FIELDS = ("ticker", "filing_type", "fiscal_period", "section", "source_url")


class FundamentalsAgent:
    name = "fundamentals"

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()

    def _context(self, sources) -> str:
        blocks = []
        for i, s in enumerate(sources, 1):
            label = f"{s['ticker']} {s['filing_type']} {s['fiscal_period']} — {s['section']}"
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
