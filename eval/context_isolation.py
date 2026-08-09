"""Milestone 4 checkpoint — prove the orchestrator's context stays bounded.

The whole reason sub-agents exist is that their verbose work stays in *their*
context and only summaries come back. That claim is testable: make the
sub-agents read progressively more (raise k) and check the orchestrator's own
prompt size barely moves.

Usage:
    python eval/context_isolation.py                    # k = 2, 6, 12, 20
    python eval/context_isolation.py -k 4 8 16

Reports, per k: how much the Fundamentals agent actually READ (chars of
retrieved passages) versus what the orchestrator INGESTED (router + synthesis
prompt chars). Passes if read volume grows several-fold while orchestrator
context grows only marginally.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from orchestrator import Orchestrator          # noqa: E402
from tools import search_filings               # noqa: E402

QUESTION = "Is NVDA expensive right now?"
TICKER = "NVDA"
# Orchestrator context may grow a little (summaries vary), but nothing like the
# read volume. Fail if it grows more than this fraction of the read growth.
MAX_CONTEXT_GROWTH_RATIO = 0.25


def read_volume(question: str, ticker: str, k: int) -> int:
    """Chars of filing text the Fundamentals agent pulls into its own context."""
    passages = search_filings(question, ticker=ticker, k=k)
    return sum(len(p["text"]) for p in passages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, nargs="+", default=[2, 6, 12, 20])
    ap.add_argument("-q", "--question", default=QUESTION)
    args = ap.parse_args()

    orch = Orchestrator()
    rows = []
    for k in args.k:
        read = read_volume(args.question, TICKER, k)
        res = orch.ask(args.question, k=k)
        rows.append((k, read, res.context_chars, res.routed,
                     sum(a.summary_chars() for a in res.sub_answers.values())))
        print(f"  k={k:<3} read={read:>7,}  orchestrator={res.context_chars:>6,}  "
              f"summaries={rows[-1][4]:>5,}  routed={','.join(res.routed)}")

    print(f"\n{'k':>4} {'sub-agent read':>15} {'orchestrator ctx':>17} {'ratio':>8}")
    for k, read, ctx, _, _ in rows:
        print(f"{k:>4} {read:>15,} {ctx:>17,} {read / max(ctx, 1):>8.1f}x")

    first, last = rows[0], rows[-1]
    read_growth = last[1] / max(first[1], 1)
    ctx_growth = last[2] / max(first[2], 1)
    print(f"\nread volume grew {read_growth:.1f}x  |  orchestrator context grew {ctx_growth:.2f}x")

    # Context must stay roughly flat: its growth should be a small fraction of
    # the read growth, not proportional to it.
    budget = 1 + (read_growth - 1) * MAX_CONTEXT_GROWTH_RATIO
    if ctx_growth <= budget:
        print(f"PASS — context isolation holds (grew {ctx_growth:.2f}x, allowed {budget:.2f}x)")
        return 0
    print(f"FAIL — orchestrator context tracks sub-agent read volume "
          f"(grew {ctx_growth:.2f}x, allowed {budget:.2f}x)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
