"""CLI over the Fundamentals sub-agent (Phase 1 Milestone 6 / Phase 2 Milestone 1).

Thin wrapper: it calls FundamentalsAgent (which returns the structured
`AgentAnswer` contract) and pretty-prints the answer with its citations.

Usage:
    export GEMINI_API_KEY=...          # or put it in .env
    python src/agent.py "What were Apple's total net sales in fiscal 2025?" --ticker AAPL
    python src/agent.py "How has Google Cloud revenue changed?" --ticker GOOGL --temporal
    python src/agent.py "..." --section "Item 1A" --dry-run   # show retrieved sources only
"""
import argparse
import textwrap

from fundamentals import FundamentalsAgent
from tools import search_filings
from llm import DEFAULT_MODEL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--ticker")
    ap.add_argument("--section", help="e.g. 'Item 1A'")
    ap.add_argument("--period", help="e.g. 'Q2 2026' or 'FY2025'")
    ap.add_argument("--type", dest="filing_type", help="10-K or 10-Q")
    ap.add_argument("--temporal", action="store_true", help="fan out per period (needs --ticker)")
    ap.add_argument("-k", type=int, default=6, help="passages to retrieve (per period if temporal)")
    ap.add_argument("--dry-run", action="store_true", help="show retrieved sources; skip the LLM")
    args = ap.parse_args()

    if args.dry_run:
        sources = search_filings(args.question, ticker=args.ticker, section=args.section,
                                 period=args.period, filing_type=args.filing_type,
                                 k=args.k, temporal=args.temporal)
        print(f"[dry run] model would be {DEFAULT_MODEL}; {len(sources)} sources retrieved:\n")
        for i, s in enumerate(sources, 1):
            print(f"[{i}] {s['ticker']} {s['filing_type']} {s['fiscal_period']} — {s['section']} [{s['chunk_kind']}]")
            print(textwrap.fill(" ".join(s["text"].split())[:300], width=96,
                                initial_indent="    ", subsequent_indent="    ") + "\n")
        return

    ans = FundamentalsAgent().answer(
        args.question, ticker=args.ticker, section=args.section, period=args.period,
        filing_type=args.filing_type, k=args.k, temporal=args.temporal)

    print("\n" + textwrap.fill(ans.answer, width=100))
    print(f"\nconfidence: {ans.confidence}"
          + ("   [insufficient evidence]" if ans.insufficient_evidence else ""))
    if ans.citations:
        print("Sources:")
        for i, c in enumerate(ans.citations, 1):
            print(f"  [{i}] {c.label()}  ({c.source_url})")


if __name__ == "__main__":
    main()
