"""Milestone 5 — measure period coverage: fan-out vs global top-k.

For cross-quarter questions, what matters isn't whether *a* relevant chunk is
retrieved, but whether *every* period is represented — otherwise a trend answer
is built on one quarter. This compares, on the eval's cross-quarter questions:

  * global top-k : plain dense retrieval, then count distinct periods covered
  * fan-out      : temporal_search, top-k within each period

with the same total retrieval budget, and reports period coverage for each.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import Retriever  # noqa: E402
from sections import section_code  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"
K_PER_PERIOD = 2


def covered_periods(hits, gold, relevant):
    """Distinct relevant periods for which a gold-section chunk was retrieved."""
    out = set()
    for h in hits:
        m = h["metadata"]
        if (m.get("ticker") == gold["ticker"]
                and section_code(m.get("section", "")) in gold["sections"]
                and m.get("fiscal_period") in relevant):
            out.add(m["fiscal_period"])
    return out


def main():
    data = json.loads(EVAL_SET.read_text())["questions"]
    cross = [q for q in data if q["type"] == "cross_quarter"]
    r = Retriever()

    g_tot = t_tot = denom = 0
    print(f"Period coverage on {len(cross)} cross-quarter questions "
          f"({K_PER_PERIOD}/period):\n")
    print(f"  {'id':<5} {'ticker':<6} {'periods':>7}  {'global':>7}  {'fanout':>7}   question")
    for q in cross:
        gold = q["gold"]
        relevant = [p for p, _ in r.periods_for(gold["ticker"], "10-Q")]
        budget = len(relevant) * K_PER_PERIOD

        g_hits = r.search(q["question"], k=budget, method="dense")
        g_cov = covered_periods(g_hits, gold, relevant)

        groups = r.temporal_search(q["question"], gold["ticker"],
                                   k_per_period=K_PER_PERIOD,
                                   filing_type="10-Q", section=gold["sections"][0])
        t_hits = [h for grp in groups for h in grp["hits"]]
        t_cov = covered_periods(t_hits, gold, relevant)

        n = len(relevant)
        g_tot += len(g_cov); t_tot += len(t_cov); denom += n
        print(f"  {q['id']:<5} {gold['ticker']:<6} {n:>7}  "
              f"{len(g_cov)}/{n:<5}  {len(t_cov)}/{n:<5}   {q['question'][:44]}")

    print(f"\n  Period coverage — global top-k: {g_tot/denom:.0%}   "
          f"fan-out: {t_tot/denom:.0%}")


if __name__ == "__main__":
    main()
