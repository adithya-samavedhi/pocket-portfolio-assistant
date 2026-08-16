"""How much of the relevant section the model actually gets to read.

Analytical questions — moat, headwinds, growth drivers, strategy — are answered
by a whole section, because the argument is spread across it. Chunk retrieval
shows a few fragments: k=6 is roughly 1.2% of a 10-K. This measures the share of
the gold section's text that reaches the model, with and without expansion.

Retrieval-only, so it costs no LLM quota.

    python eval/run_section_coverage.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import Retriever  # noqa: E402
from sections import section_code  # noqa: E402
from tools import SECTION_BUDGET_CHARS  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"
ANALYTICAL = {"moat", "headwind", "growth_driver", "risk_linkage", "multi_hop"}
K = 6


def gold_chars(r, hits, gold):
    """Chars shown from a chunk whose section is one the question is about."""
    return sum(len(h["document"]) for h in hits
               if section_code(h["metadata"].get("section", "")) in gold["sections"]
               and h["metadata"].get("ticker") in (gold.get("tickers") or [gold.get("ticker")]))


def main():
    qs = [q for q in json.loads(EVAL_SET.read_text())["questions"]
          if q.get("gold") and q["type"] in ANALYTICAL]
    r = Retriever()

    print(f"Evidence shown on {len(qs)} analytical questions "
          f"(k={K}, expansion budget {SECTION_BUDGET_CHARS:,} chars):\n")
    print(f"  {'id':<6} {'type':<14} {'chunks':>9} {'expanded':>10} {'gain':>7}   question")

    tot_c = tot_e = 0
    for q in qs:
        base = r.search(q["question"], k=K)
        exp = r.expand_sections(base, budget_chars=SECTION_BUDGET_CHARS)
        c, e = gold_chars(r, base, q["gold"]), gold_chars(r, exp, q["gold"])
        tot_c, tot_e = tot_c + c, tot_e + e
        gain = f"{e / c:.1f}x" if c else "  n/a"
        print(f"  {q['id']:<6} {q['type']:<14} {c:>9,} {e:>10,} {gain:>7}   {q['question'][:40]}")

    print(f"\n  total relevant text shown — chunks: {tot_c:,} chars   "
          f"expanded: {tot_e:,} chars   ({tot_e / max(tot_c, 1):.1f}x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
