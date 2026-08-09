"""Milestone 5 — measure routing accuracy (and reconciliation on 'both').

Usage:
    python eval/run_routing.py                 # routing only: 1 LLM call/question
    python eval/run_routing.py --reconcile     # also run the 'both' questions end
                                               # to end (3 calls each) and check the
                                               # final answer cites BOTH sources
    python eval/run_routing.py --limit 8       # smoke-test on a subset

Routing is scored as an exact set match: consulting an extra specialist is a
miss, not a partial credit, because a spurious sub-agent call costs latency and
tokens and drags irrelevant material into the synthesis.

Reconciliation is scored structurally, not by vibes: on a 'both' question the
final answer must cite at least one filings source AND at least one market-data
source. An answer that quietly drops one specialist fails.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from orchestrator import Orchestrator          # noqa: E402
from trace import Trace                        # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "routing_set.json"


def label(agents) -> str:
    """Bucket a set of agents into the four eval classes."""
    s = set(agents)
    if not s:
        return "neither"
    if s == {"fundamentals", "technicals"}:
        return "both"
    return next(iter(s))


def source_kind(citation) -> str:
    return "technicals" if citation.filing_type.startswith("market-data") else "fundamentals"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reconcile", action="store_true",
                    help="run 'both' questions end to end and check they cite both sources")
    ap.add_argument("--limit", type=int, help="only evaluate the first N questions")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    questions = json.loads(EVAL_SET.read_text())["questions"]
    if args.limit:
        questions = questions[:args.limit]

    orch = Orchestrator()
    by_class = defaultdict(lambda: {"n": 0, "hit": 0})
    misses = []
    errors = []

    print(f"Routing {len(questions)} questions with {orch.llm.model}...\n")
    for i, item in enumerate(questions, 1):
        gold = set(item["agents"])
        gold_class = label(gold)
        tr = Trace(item["q"])
        # A quota failure partway through must not throw away the questions
        # already scored — free-tier daily limits make that a routine event.
        try:
            route = orch.route(item["q"], tr)
        except Exception as e:                       # noqa: BLE001
            errors.append((item["q"], f"{type(e).__name__}"))
            print(f"  ERR  [{i:>2}] {item['q'][:62]:<62} {type(e).__name__}")
            continue
        got = set(route["agents"])

        hit = got == gold
        by_class[gold_class]["n"] += 1
        by_class[gold_class]["hit"] += hit
        if not hit:
            misses.append((item["q"], gold_class, label(got), route["reason"]))
        if args.verbose or not hit:
            mark = "ok " if hit else "MISS"
            print(f"  {mark} [{i:>2}] {item['q'][:62]:<62} gold={gold_class:<12} got={label(got)}")

    total = sum(c["n"] for c in by_class.values())
    hits = sum(c["hit"] for c in by_class.values())
    if not total:
        print("\nNo questions scored — every call failed. "
              f"({errors[0][1] if errors else 'unknown'})")
        return 1
    print(f"\n{'class':<14} {'n':>3} {'correct':>8} {'accuracy':>9}")
    for cls in ("fundamentals", "technicals", "both", "neither"):
        c = by_class.get(cls)
        if c:
            print(f"{cls:<14} {c['n']:>3} {c['hit']:>8} {c['hit'] / c['n']:>8.0%}")
    print(f"{'OVERALL':<14} {total:>3} {hits:>8} {hits / total:>8.0%}")

    if errors:
        print(f"\n{len(errors)} question(s) could not be scored (excluded from the "
              f"numbers above): {errors[0][1]}. Re-run to resume — completed calls "
              f"are cached, so only the failures cost quota.")

    if misses:
        print("\nMisrouted:")
        for q, gold, got, reason in misses:
            print(f"  - {q}\n      gold={gold}  got={got}\n      router said: {reason}")

    if args.reconcile:
        both = [i for i in questions if label(i["agents"]) == "both"]
        print(f"\nReconciliation check on {len(both)} 'both' questions "
              f"(full runs — this is the slow part)...")
        passed = 0
        for item in both:
            res = orch.ask(item["q"])
            kinds = {source_kind(c) for c in res.answer.citations}
            ok = kinds == {"fundamentals", "technicals"}
            passed += ok
            print(f"  {'ok  ' if ok else 'FAIL'} {item['q'][:58]:<58} "
                  f"cited={','.join(sorted(kinds)) or 'none':<26} ctx={res.context_chars}")
        print(f"\nreconciliation: {passed}/{len(both)} ({passed / max(len(both), 1):.0%}) "
              f"cited both sources")

    return 0 if hits == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
