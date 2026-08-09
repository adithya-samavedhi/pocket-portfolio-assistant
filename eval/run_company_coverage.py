"""Company coverage on comparative questions: global top-k vs per-company fan-out.

A question like "compare how Microsoft and Amazon describe cloud growth" needs
passages from BOTH filings. Global top-k has no reason to supply that — one
company's language can simply score better and take every slot, and the answer
then compares a company with itself.

This is the same failure temporal fan-out fixed for periods (89% -> 100% period
coverage), asked of companies instead. Retrieval-only, so it needs no LLM quota.

    python eval/run_company_coverage.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import Retriever  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"
K = 10


def covered(metas, tickers):
    """How many of the named companies actually appear in the results."""
    seen = {m["ticker"] for m in metas}
    return len(seen & set(tickers))


def main():
    qs = [q for q in json.loads(EVAL_SET.read_text())["questions"]
          if (q.get("gold") or {}).get("tickers")]
    if not qs:
        print("no multi-company questions in the eval set")
        return 0

    r = Retriever()
    print(f"Company coverage on {len(qs)} comparative questions (k={K}):\n")
    print(f"  {'id':<6} {'companies':<14} {'global':>8} {'fan-out':>9}   question")

    tot_g = tot_f = tot_n = 0
    for q in qs:
        tickers = q["gold"]["tickers"]
        glob = r.search(q["question"], k=K)
        g = covered([h["metadata"] for h in glob], tickers)

        # Fan out: an equal share of the budget per named company.
        per = max(1, K // len(tickers))
        fan = []
        for t in tickers:
            fan += r.search(q["question"], k=per, where={"ticker": t})
        f = covered([h["metadata"] for h in fan], tickers)

        tot_g, tot_f, tot_n = tot_g + g, tot_f + f, tot_n + len(tickers)
        flag = "" if g == len(tickers) else "  <- global missed one"
        print(f"  {q['id']:<6} {','.join(tickers):<14} {g}/{len(tickers):<6} "
              f"{f}/{len(tickers):<7}   {q['question'][:44]}{flag}")

    print(f"\n  Company coverage — global top-k: {tot_g/tot_n:.0%}   "
          f"per-company fan-out: {tot_f/tot_n:.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
