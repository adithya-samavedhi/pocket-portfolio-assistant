"""Orchestrator: router + fan-out / fan-in (Phase 2, Milestone 4).

The coordinator, and the heart of Phase 2. Three steps:

  1. ROUTE    — an LLM classifier decides which specialists the question needs.
  2. FAN OUT  — the chosen sub-agents run in parallel (latency stacks otherwise).
  3. FAN IN   — their structured summaries are synthesized into one cited answer
                that reconciles them.

**Context isolation is the whole point.** The orchestrator never sees a filing
passage or a price series — only `AgentAnswer` summaries. Its own context is
therefore bounded by the number of sub-agents, not by how much they read. That
property is enforced here (over-budget summaries are truncated and traced) and
measured by eval/context_isolation.py.
"""
import textwrap
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from contracts import SUMMARY_CHAR_BUDGET, AgentAnswer, Citation, Confidence, renumber
from fundamentals import FundamentalsAgent
from llm import LLM
from technicals import TICKERS, TechnicalsAgent
from trace import Trace

# What the router picks from. The descriptions are the routing prompt's only
# knowledge of each specialist, so they carry the routing quality.
SPECIALISTS = {
    "fundamentals": "SEC filings (10-K/10-Q): revenue, segments, margins, risk "
                    "factors, MD&A commentary, business descriptions, and how any "
                    "of that changed across quarters. The source of record for "
                    "what a company reported or said.",
    "technicals": "Market data: current price, valuation multiples (P/E, P/S, "
                  "market cap), momentum over 1 month to 1 year, moving averages, "
                  "position in the 52-week range, drawdown, volatility. The source "
                  "for what the stock has done and what it costs today.",
}

ROUTER_SYSTEM = """You route finance questions to specialist sub-agents.

Available specialists:
{specialists}

Return a single JSON object with exactly these keys:
  "agents": array of specialist names to consult. Use BOTH when the question
            needs the company's reported business AND its market price/valuation
            (e.g. "is this stock expensive?" — the multiple is market data, but
            whether it is justified is in the filings). Use ONE when only that
            source can answer it. Use [] when NEITHER can (general knowledge,
            macro, personal advice, or a company outside coverage).
  "ticker": the covered ticker the question is about, or null. Covered: {tickers}.
  "temporal": true only if the question asks how something CHANGED over quarters.
  "analytical": true if answering well needs a WHOLE section of a filing read and
            weighed — competitive position or moat, what the company says it
            competes on, the risks or headwinds it discloses, what it says will
            drive growth, its strategy. False for a lookup of a specific figure,
            date or fact, which a few passages answer better.
  "subquestions": an object mapping each chosen specialist to the question IT
            should be asked. Rewrite the user's question into what that source
            can actually answer — a filings agent cannot answer "is it
            expensive?", but it can answer "what were revenue growth and margins
            in recent quarters?". Keep the user's intent; change only the framing.
  "reason": one short sentence on why you routed it that way.

Route on what the question needs, not on the words it uses.

Return [] even when a covered ticker is named, if the question asks for:
- a BUY/SELL/HOLD recommendation or what someone should do ("should I buy X?"),
- a PREDICTION about future prices or events ("will X go up next week?").
No source answers those, so consulting a specialist only wastes a call. Note that
"is X expensive?" is NOT advice — it is a valuation question, and it routes to both.

Output JSON only."""

SYNTH_SYSTEM = """You are the orchestrator in a multi-agent finance system. \
Specialist sub-agents have each answered from their own source. Synthesize ONE \
grounded answer.

Return a single JSON object with exactly these keys:
  "answer": the synthesized answer (<= 150 words), citing sources inline as [1],
            [2] using the numbers in the source list.
  "cited": array of the source numbers you actually used.
  "confidence": one of "low", "medium", "high".
  "insufficient_evidence": true if the specialists could not answer the question.

Rules:
- Use ONLY facts the specialists reported. Never add figures of your own.
- RECONCILE the views rather than concatenating them: if the filings view and the
  market view point different ways, say so explicitly and explain the tension.
  Do not present a contradiction as agreement.
- If a specialist reported insufficient evidence, or flagged something as
  unavailable, carry that caveat into the answer instead of dropping it.
- Attribute where it matters ("the filings show...", "the price data shows...").
- If you use a specialist's finding, CITE it. Every specialist that reported
  evidence must appear in "cited" — dropping one silently turns a two-sided
  answer into a one-sided one. Only omit a specialist that reported nothing.
- Describe; do not give buy/sell advice.

Output JSON only, no prose around it."""


@dataclass
class OrchestratorResult:
    """The final answer plus everything needed to debug how it was produced."""
    answer: AgentAnswer
    routed: list[str] = field(default_factory=list)
    route_reason: str = ""
    ticker: Optional[str] = None
    sub_answers: dict[str, AgentAnswer] = field(default_factory=dict)
    trace: dict = field(default_factory=dict)
    # Chars of prompt the orchestrator itself ingested (router + synthesis).
    # The context-isolation number: it must not grow with sub-agent read volume.
    context_chars: int = 0


class Orchestrator:
    def __init__(self, llm: Optional[LLM] = None, fundamentals=None, technicals=None):
        self.llm = llm or LLM()
        self.agents = {
            "fundamentals": fundamentals or FundamentalsAgent(llm=self.llm),
            "technicals": technicals or TechnicalsAgent(llm=self.llm),
        }

    # --- 1. route ---
    def route(self, question: str, tr: Trace) -> dict:
        system = ROUTER_SYSTEM.format(
            specialists="\n".join(f"- {n}: {d}" for n, d in SPECIALISTS.items()),
            tickers=", ".join(sorted(TICKERS)),
        )
        with tr.timer("route") as t:
            data = self.llm.complete_json(system, f"Question: {question}")
            agents = [a for a in (data.get("agents") or []) if a in self.agents]
            # Fall back to the raw question if the router omitted a sub-question.
            subq = data.get("subquestions") or {}
            subquestions = {a: str(subq.get(a) or question) for a in agents}
            t.add(agents=",".join(agents) or "none", ticker=data.get("ticker"),
                  analytical=bool(data.get("analytical")),
                  reason=data.get("reason", "")[:80])
        return {"agents": agents, "ticker": data.get("ticker"),
                "temporal": bool(data.get("temporal")),
                "analytical": bool(data.get("analytical")),
                "reason": data.get("reason", ""),
                "subquestions": subquestions,
                "prompt_chars": len(system) + len(question)}

    # --- 2. fan out ---
    def _call(self, name: str, question: str, route: dict, k: int, tr: Trace) -> AgentAnswer:
        # Each specialist gets the question rewritten for its own source.
        asked = route["subquestions"].get(name, question)
        with tr.timer("subagent", agent=name, asked=asked[:60]) as t:
            try:
                if name == "fundamentals":
                    ans = self.agents[name].answer(
                        asked, ticker=route["ticker"], k=k,
                        temporal=route["temporal"] and bool(route["ticker"]),
                        expand_section=route.get("analytical", False))
                else:
                    ans = self.agents[name].answer(asked, ticker=route["ticker"])
            except Exception as e:                       # noqa: BLE001 - one agent down != run down
                t.add(failed=True)
                return AgentAnswer(agent=name, insufficient_evidence=True,
                                   answer=f"{name} sub-agent failed: {type(e).__name__}: {e}")
            t.add(chars=ans.summary_chars(), cites=len(ans.citations),
                  conf=ans.confidence, insufficient=ans.insufficient_evidence)
        return ans

    def fan_out(self, names: list[str], question: str, route: dict, k: int,
                tr: Trace) -> dict[str, AgentAnswer]:
        if not names:
            return {}
        if len(names) == 1:
            return {names[0]: self._call(names[0], question, route, k, tr)}
        # Independent calls — run them concurrently so latency doesn't stack.
        with ThreadPoolExecutor(max_workers=len(names)) as pool:
            futures = {n: pool.submit(self._call, n, question, route, k, tr) for n in names}
            return {n: f.result() for n, f in futures.items()}

    # --- 3. fan in ---
    def _synth_context(self, subs: dict[str, AgentAnswer], tr: Trace):
        """Render sub-agent summaries into a numbered source list.

        This is the context-isolation boundary: only `answer` text and citation
        *labels* cross it. Over-budget summaries are truncated here so one
        misbehaving sub-agent cannot blow the orchestrator's context.
        """
        blocks, citations = [], []
        for name, ans in subs.items():
            if not ans.within_budget():
                tr.span("budget_exceeded", agent=name, chars=ans.summary_chars(),
                        budget=SUMMARY_CHAR_BUDGET)
                ans = ans.model_copy(update={"answer": ans.answer[:SUMMARY_CHAR_BUDGET]})
            lines = [f"### {name} (confidence: {ans.confidence}"
                     + (", INSUFFICIENT EVIDENCE" if ans.insufficient_evidence else "") + ")",
                     ans.answer]
            for c in ans.citations:
                citations.append(c)
                lines.append(f"  [{len(citations)}] {c.label()}")
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks), citations

    def synthesize(self, question: str, subs: dict[str, AgentAnswer],
                   tr: Trace) -> tuple[AgentAnswer, int]:
        context, citations = self._synth_context(subs, tr)
        user = f"Specialist findings:\n{context}\n\nQuestion: {question}"
        with tr.timer("synthesize") as t:
            data = self.llm.complete_json(SYNTH_SYSTEM, user)
            t.add(context_chars=len(user), sources=len(citations))

        # Dedupe the cited sources, remembering where each original source
        # number ended up so the inline markers can be renumbered to match.
        cited, seen, remap = [], {}, {}
        for n in (data.get("cited") or []):
            if isinstance(n, int) and 1 <= n <= len(citations):
                c = citations[n - 1]
                key = c.source_url + c.label()
                if key not in seen:
                    cited.append(c)
                    seen[key] = len(cited)
                remap[n] = seen[key]

        confidence: Confidence = data.get("confidence", "low")
        if confidence not in ("low", "medium", "high"):
            confidence = "low"

        answer = AgentAnswer(
            agent="orchestrator",
            answer=renumber(str(data.get("answer", "")).strip(), remap),
            citations=cited,
            confidence=confidence,
            insufficient_evidence=bool(data.get("insufficient_evidence", False)),
        )
        return answer, len(SYNTH_SYSTEM) + len(user)

    # --- the run ---
    def ask(self, question: str, k: int = 6, verbose: bool = False,
            on_span=None) -> OrchestratorResult:
        tr = Trace(question, verbose=verbose, on_span=on_span)
        route = self.route(question, tr)

        if not route["agents"]:
            answer = AgentAnswer(
                agent="orchestrator", insufficient_evidence=True,
                answer="No specialist covers this question. I can answer from SEC "
                       "filings (fundamentals) or market data (technicals) for "
                       "AAPL, AMZN, GOOGL, MSFT, and NVDA.")
            data = tr.finish(routed=[], answer=answer.answer, context_chars=route["prompt_chars"])
            return OrchestratorResult(answer=answer, routed=[], route_reason=route["reason"],
                                      ticker=route["ticker"], trace=data,
                                      context_chars=route["prompt_chars"])

        subs = self.fan_out(route["agents"], question, route, k, tr)
        answer, synth_chars = self.synthesize(question, subs, tr)
        context_chars = route["prompt_chars"] + synth_chars

        data = tr.finish(routed=route["agents"], route_reason=route["reason"],
                         ticker=route["ticker"], temporal=route["temporal"],
                         confidence=answer.confidence, context_chars=context_chars,
                         sub_summary_chars={n: a.summary_chars() for n, a in subs.items()})
        return OrchestratorResult(answer=answer, routed=route["agents"],
                                  route_reason=route["reason"], ticker=route["ticker"],
                                  sub_answers=subs, trace=data, context_chars=context_chars)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Ask the multi-agent orchestrator a question.")
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=6, help="passages per fundamentals retrieval")
    ap.add_argument("-v", "--verbose", action="store_true", help="stream the trace to stderr")
    args = ap.parse_args()

    res = Orchestrator().ask(args.question, k=args.k, verbose=args.verbose)

    print(f"\nrouted to: {', '.join(res.routed) or 'none'}"
          + (f"  ({res.route_reason})" if res.route_reason else ""))
    print("\n" + textwrap.fill(res.answer.answer, width=100))
    print(f"\nconfidence: {res.answer.confidence}"
          + ("   [insufficient evidence]" if res.answer.insufficient_evidence else ""))
    if res.answer.citations:
        print("Sources:")
        for i, c in enumerate(res.answer.citations, 1):
            print(f"  [{i}] {c.label()}  ({c.source_url})")
    print(f"\norchestrator context: {res.context_chars} chars"
          f" | sub-agent summaries: "
          + ", ".join(f"{n}={a.summary_chars()}" for n, a in res.sub_answers.items())
          + f" | trace: data/traces/{res.trace.get('trace_id')}.json")


if __name__ == "__main__":
    main()
