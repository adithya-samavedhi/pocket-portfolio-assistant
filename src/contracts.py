"""Sub-agent hand-off contract (Phase 2).

`AgentAnswer` is the ONE structure every specialist sub-agent returns and the
orchestrator consumes. It is a compact *summary* — an answer, the sources it
used (metadata only, never the raw passage text), a self-reported confidence,
and an explicit "not enough evidence" flag. Keeping raw passages out of this
object is what keeps the orchestrator's context bounded.

Locked before the orchestrator is built: changing this schema later means
touching every sub-agent.
"""
import re
from typing import Dict, List, Literal

from pydantic import BaseModel, Field, model_validator

Confidence = Literal["low", "medium", "high"]

# Soft budget for a summary handed back to the orchestrator. Sub-agents aim
# under this; the orchestrator can assert on it to catch raw-dump regressions.
#
# 2400 fits a ~200-word answer plus its citations. Raised from 1500 when the
# answering prompt was asked for per-period figures rather than a one-line
# summary — the old cap silently truncated exactly the detail we wanted. It is
# still ~2% of a filing, so the point of the budget (summaries, never dumps)
# holds; eval/context_isolation.py is what proves it.
SUMMARY_CHAR_BUDGET = 2400


class Citation(BaseModel):
    """A source the sub-agent actually used — metadata only, no passage text."""
    ticker: str
    filing_type: str
    fiscal_period: str
    section: str
    source_url: str

    def label(self) -> str:
        return f"{self.ticker} {self.filing_type} {self.fiscal_period} — {self.section}"


class AgentAnswer(BaseModel):
    """What a specialist sub-agent returns to the orchestrator."""
    agent: str                                  # which sub-agent produced this
    answer: str                                 # the synthesized answer text
    citations: List[Citation] = Field(default_factory=list)
    confidence: Confidence = "low"
    insufficient_evidence: bool = False

    @model_validator(mode="after")
    def _coherent(self):
        """Keep the flags from contradicting each other.

        Models routinely return `insufficient_evidence=true` alongside
        `confidence="high"` (observed in the M5 reconciliation run). The
        orchestrator weighs confidence when synthesizing, so an incoherent pair
        actively misleads it: "I found nothing, and I'm sure of it" reads as a
        strong finding. Nothing found means low confidence, and no citations.
        """
        if self.insufficient_evidence:
            object.__setattr__(self, "confidence", "low")
            object.__setattr__(self, "citations", [])
        return self

    def summary_chars(self) -> int:
        """Size of what the orchestrator will actually ingest."""
        return len(self.answer) + sum(len(c.label()) + len(c.source_url) for c in self.citations)

    def within_budget(self, budget: int = SUMMARY_CHAR_BUDGET) -> bool:
        return self.summary_chars() <= budget


def renumber(answer: str, remap: Dict[int, int]) -> str:
    """Rewrite inline [n] markers to match the deduped citation list.

    A model cites the source numbers it was shown, but deduping renumbers the
    final list — without this, an answer can cite [7] under a 5-source list.
    Handles both "[1], [3]" and "[1, 3]". Markers whose source did not survive
    are dropped rather than left pointing at the wrong row.
    """
    def sub(m):
        nums = [remap.get(int(x)) for x in re.findall(r"\d+", m.group(1))]
        kept = sorted({n for n in nums if n})
        return ", ".join(f"[{n}]" for n in kept) if kept else ""

    return re.sub(r"\[([\d\s,]+)\]", sub, answer).replace("  ", " ").strip()
