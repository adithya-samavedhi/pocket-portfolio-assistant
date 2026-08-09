"""Measure retrieval quality against the Milestone 3 eval set.

Usage:
    python eval/run_eval.py            # recall@5, recall@10, MRR over the eval set
    python eval/run_eval.py -k 20      # also report recall@20

A question scores a hit@k if any of the top-k retrieved chunks matches the gold
ticker AND one of the gold sections (item code before the colon) AND, when the
question pins a period, the gold period. Retrieval is UNFILTERED — pure semantic
search over the whole corpus — because that is the quality Milestone 4 improves.

Negatives (gold=null) can't be scored by recall; we report what retrieval pulled
back and the top-1 similarity, to calibrate an "I don't know" threshold later.
"""
import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from retrieval import Retriever  # noqa: E402
from sections import section_code  # noqa: E402
from tools import DEFAULT_TABLE_QUOTA  # noqa: E402

EVAL_SET = Path(__file__).resolve().parent / "eval_set.json"


def is_hit(meta, gold) -> bool:
    # Comparative questions span companies: any of the named tickers counts,
    # since a single chunk can only ever come from one filing.
    tickers = gold.get("tickers") or [gold["ticker"]]
    if meta.get("ticker") not in tickers:
        return False
    if section_code(meta.get("section", "")) not in gold["sections"]:
        return False
    if "period" in gold and meta.get("fiscal_period") != gold["period"]:
        return False
    return True


def first_hit_rank(metas, gold):
    for rank, meta in enumerate(metas, 1):
        if is_hit(meta, gold):
            return rank
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-k", type=int, default=10, help="max k to retrieve/report")
    ap.add_argument("--method", default="dense", choices=["dense", "bm25", "hybrid", "rerank"])
    ap.add_argument("--table-quota", type=int, default=None,
                    help="slots reserved for table chunks (default: the production value)")
    args = ap.parse_args()
    quota = DEFAULT_TABLE_QUOTA if args.table_quota is None else args.table_quota
    ks = sorted({1, 3, 5, 10, args.k})

    data = json.loads(EVAL_SET.read_text())["questions"]
    answerable = [q for q in data if q.get("gold")]
    negatives = [q for q in data if not q.get("gold")]

    retriever = Retriever()

    by_type = defaultdict(list)
    rr_sum, misses = 0.0, []
    print(f"[method={args.method}] top-{max(ks)} (unfiltered), "
          f"{len(answerable)} answerable questions...\n")

    table_rows = []
    for q in answerable:
        hits = retriever.search(q["question"], k=max(ks), method=args.method,
                                table_quota=quota)
        metas = [h["metadata"] for h in hits]
        rank = first_hit_rank(metas, q["gold"])
        rr_sum += (1.0 / rank) if rank else 0.0
        row = {"id": q["id"], "rank": rank}
        by_type[q["type"]].append(row)
        # Questions whose answer is a figure: did any table actually reach the
        # model? Section-level recall cannot see this, which is how "the filings
        # don't disclose that" slipped through while the table sat in the corpus.
        if q.get("expect_table"):
            # Count across everything retrieved: the whole set is handed to the
            # model, and tables are deliberately placed last (nearest the question).
            got = sum(1 for m in metas if m.get("chunk_kind") == "table")
            table_rows.append((q["id"], got))
        if rank is None or rank > 5:
            top = metas[0]
            misses.append((q["id"], q["type"], rank, q["question"][:52],
                           f"{top.get('ticker')} {section_code(top.get('section',''))} {top.get('fiscal_period')}"))

    def recall(rows, k):
        hits = sum(1 for r in rows if r["rank"] and r["rank"] <= k)
        return hits / len(rows) if rows else 0.0

    if table_rows:
        with_tab = sum(1 for _, g in table_rows if g)
        print(f"Table coverage (numeric questions, quota={quota}): "
              f"{with_tab}/{len(table_rows)} got at least one table"
              + ("" if with_tab == len(table_rows)
                 else "  -> missing: " + ", ".join(i for i, g in table_rows if not g)))
        print()

    print("By question type:")
    print(f"  {'type':<14} {'n':>3} " + " ".join(f'R@{k:<3}' for k in ks))
    order = ["factual", "section", "cross_quarter", "segment", "trend",
             "moat", "headwind", "growth_driver", "multi_hop", "comparative",
             "quantitative", "risk_linkage", "accounting", "capital_allocation",
             "concentration", "legal"]
    for t in order + [t for t in by_type if t not in order]:
        rows = by_type.get(t, [])
        if rows:
            print(f"  {t:<14} {len(rows):>3} " +
                  " ".join(f'{recall(rows,k):>4.0%}' for k in ks))
    allrows = [r for rows in by_type.values() for r in rows]
    print(f"  {'OVERALL':<14} {len(allrows):>3} " +
          " ".join(f'{recall(allrows,k):>4.0%}' for k in ks))
    print(f"\n  MRR@{max(ks)} = {rr_sum/len(allrows):.3f}")

    if misses:
        print(f"\nMisses / rank>5 ({len(misses)}):")
        for qid, t, rank, text, top in misses:
            print(f"  {qid} [{t:<12}] rank={str(rank):<4} got_top1=[{top:<22}]  {text}")

    print(f"\nNegatives ({len(negatives)}) — retrieval can't score these; top-1 shown:")
    for q in negatives:
        m = retriever.search(q["question"], k=1, method=args.method)[0]["metadata"]
        print(f"  {q['id']}  top1=[{m.get('ticker')} {section_code(m.get('section',''))} "
              f"{m.get('fiscal_period')}]  {q['question'][:50]}")


if __name__ == "__main__":
    main()
