"""Technicals / valuation sub-agent (Phase 2, Milestone 3).

The second specialist, and the one that proves the pattern generalizes past a
single data source. It answers "is it expensive / how is it trending" from the
market-data tools and returns the same `AgentAnswer` contract the Fundamentals
sub-agent does.

Division of labour: every number is computed in `marketdata` (deterministic,
cached, full series stays there); the LLM only interprets an already-small
metrics block. It never does arithmetic, and it never sees a price series.
"""
import re
from typing import Optional

import marketdata
from contracts import AgentAnswer, Citation, Confidence
from llm import LLM

# The covered universe, plus the names people actually type.
TICKERS = {"GOOGL", "NVDA", "MSFT", "AAPL", "AMZN"}
ALIASES = {
    "GOOGLE": "GOOGL", "ALPHABET": "GOOGL", "GOOG": "GOOGL",
    "NVIDIA": "NVDA", "MICROSOFT": "MSFT",
    "APPLE": "AAPL", "AMAZON": "AMZN",
}

SYSTEM = """You are the Technicals specialist in a multi-agent finance system. \
You answer "is it expensive / how is it trending" questions about a single stock \
using ONLY the numbered market-data sources given to you.

Every figure you need is already computed. Do NOT do arithmetic, and do NOT \
introduce any number that is not in the sources.

Return a single JSON object with exactly these keys:
  "answer": a concise answer (<= 120 words) citing sources inline as [1], [2].
            Always state the as-of date. Quote the specific figures that drive
            your read (momentum, position in the 52-week range, distance from
            the moving averages, P/E if present).
  "cited": array of the source numbers you actually used, e.g. [1, 3].
  "confidence": one of "low", "medium", "high".
  "insufficient_evidence": true if the sources do not answer the question.

Interpretation rules:
- "Expensive versus its own history" can only be answered on PRICE and TREND \
here unless a valuation source is present; when pe_history_available is false, \
say plainly that no historical P/E series was available, and do not imply the \
current P/E is high or low relative to the stock's past.
- A current P/E is a point-in-time figure, not a historical comparison.
- Describe what the numbers show; do not give buy/sell advice or price targets.

Output JSON only, no prose around it."""

# Where each tool's numbers come from, for the citation trail.
_YAHOO = "https://finance.yahoo.com/quote/{ticker}"
_FINNHUB = "https://finnhub.io/api/v1/quote?symbol={ticker}"


def resolve_ticker(question: str) -> Optional[str]:
    """Pull a covered ticker out of the question, by symbol or company name."""
    for word in re.findall(r"[A-Za-z]{2,}", question):
        upper = word.upper()
        if upper in TICKERS:
            return upper
        if upper in ALIASES:
            return ALIASES[upper]
    return None


class TechnicalsAgent:
    name = "technicals"

    def __init__(self, llm: Optional[LLM] = None):
        self.llm = llm or LLM()

    def _sources(self, ticker: str) -> list[dict]:
        """Call the market-data tools; each result becomes one numbered source.

        A tool that fails is skipped rather than fatal — a valuation outage
        should still leave a usable trend answer.
        """
        sources = []
        for section, fetch in (("technicals", marketdata.get_technicals),
                               ("valuation", marketdata.get_valuation),
                               ("quote", marketdata.get_quote)):
            try:
                data = fetch(ticker)
            except Exception as e:                      # noqa: BLE001 - degrade, don't crash
                print(f"[technicals] {section} unavailable: {e}")
                continue
            if data.get("note") == "not enough history":
                continue
            sources.append({"section": section, "data": data})
        return sources

    @staticmethod
    def _fmt(v):
        """Round vendor float noise so the model can't quote 'P/E of 33.577213'."""
        if isinstance(v, float):
            return round(v, 2)
        if isinstance(v, dict):
            return {k: TechnicalsAgent._fmt(x) for k, x in v.items()}
        return v

    def _context(self, sources) -> str:
        blocks = []
        for i, s in enumerate(sources, 1):
            fields = ", ".join(f"{k}={self._fmt(v)}" for k, v in s["data"].items()
                               if k not in ("ticker", "source") and v is not None)
            blocks.append(f"[{i}] ({s['data']['ticker']} market-data — {s['section']})\n{fields}")
        return "\n\n".join(blocks)

    def _citation(self, s) -> Citation:
        """Fit a market-data source to the locked (filings-shaped) Citation."""
        data = s["data"]
        url = (_FINNHUB if data.get("source") == "finnhub" else _YAHOO)
        return Citation(
            ticker=data["ticker"],
            filing_type=f"market-data ({data.get('source', 'unknown')})",
            fiscal_period=f"as of {data.get('as_of', 'n/a')}",
            section=s["section"],
            source_url=url.format(ticker=data["ticker"]),
        )

    def answer(self, question: str, ticker: str = None) -> AgentAnswer:
        ticker = (ticker or resolve_ticker(question) or "").upper()
        if not ticker:
            return AgentAnswer(
                agent=self.name, insufficient_evidence=True,
                answer="No covered ticker found in the question. Covered: "
                       + ", ".join(sorted(TICKERS)) + ".")

        sources = self._sources(ticker)
        if not sources:
            return AgentAnswer(agent=self.name, insufficient_evidence=True,
                               answer=f"No market data available for {ticker}.")

        data = self.llm.complete_json(
            SYSTEM, f"Sources:\n{self._context(sources)}\n\nQuestion: {question}")

        citations = [self._citation(sources[n - 1]) for n in (data.get("cited") or [])
                     if isinstance(n, int) and 1 <= n <= len(sources)]

        confidence: Confidence = data.get("confidence", "low")
        if confidence not in ("low", "medium", "high"):
            confidence = "low"

        return AgentAnswer(
            agent=self.name,
            answer=str(data.get("answer", "")).strip(),
            citations=citations,
            confidence=confidence,
            insufficient_evidence=bool(data.get("insufficient_evidence", False)),
        )
