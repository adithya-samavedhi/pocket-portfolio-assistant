"""Query the filings collection, with optional metadata filtering.

Usage:
    python src/query.py "what are the main risk factors?"
    python src/query.py "revenue by segment" -k 5 --type 10-Q --period "Q2 2026"
    python src/query.py "supply chain risks" --section "Item 1A"
    python src/query.py "data center revenue" --method rerank   # experiment
    python src/query.py "how has cloud revenue changed" --temporal --ticker GOOGL

Retrieval goes through src/retrieval.py (shared with the eval). Section labels
and chunk kind (text/table) are shown. Default method is `dense` — see the
Milestone 4 note in retrieval.py for why hybrid/rerank aren't the default.
"""
import argparse
import textwrap

from retrieval import Retriever

DEFAULT_QUESTION = "what are the main risk factors?"


def build_where(args):
    """Compose a metadata filter from CLI flags (equality clauses)."""
    clauses = []
    if args.ticker:
        clauses.append({"ticker": args.ticker})
    if args.type:
        clauses.append({"filing_type": args.type})
    if args.period:
        clauses.append({"fiscal_period": args.period})
    if args.kind:
        clauses.append({"chunk_kind": args.kind})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _print_chunk(rank, meta, doc):
    print(f"  [{rank}] {meta['filing_type']} {meta['fiscal_period']}  |  "
          f"{meta['section']}  [{meta['chunk_kind']}]")
    snippet = textwrap.shorten(doc.replace("\n", " "), width=320, placeholder=" ...")
    print(textwrap.fill(snippet, width=96, initial_indent="      ", subsequent_indent="      "))


def run_temporal(retriever, args):
    if not args.ticker:
        raise SystemExit("--temporal requires --ticker (temporal reasoning is per-company)")
    groups = retriever.temporal_search(
        args.question, ticker=args.ticker, k_per_period=args.k,
        filing_type=args.type, section=args.section)
    print(f'\nQ: "{args.question}"  [temporal fan-out: {args.ticker}, '
          f'{len(groups)} periods, {args.k}/period]\n')
    for g in groups:
        print(f"=== {g['period']}  ({g['report_date']}) ===")
        if not g["hits"]:
            print("      (no matching chunk)")
        for i, h in enumerate(g["hits"], 1):
            _print_chunk(i, h["metadata"], h["document"])
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    ap.add_argument("-k", type=int, default=5, help="number of chunks to return")
    ap.add_argument("--method", default="dense", choices=["dense", "bm25", "hybrid", "rerank"])
    ap.add_argument("--ticker")
    ap.add_argument("--type", help="filing type, e.g. 10-K or 10-Q")
    ap.add_argument("--period", help="fiscal period, e.g. 'Q2 2026' or 'FY2025'")
    ap.add_argument("--section", help="substring match on section label, e.g. 'Item 1A'")
    ap.add_argument("--kind", choices=["text", "table"], help="restrict to text or table chunks")
    ap.add_argument("--temporal", action="store_true",
                    help="fan out per period (needs --ticker); -k is per-period")
    args = ap.parse_args()

    retriever = Retriever()

    if args.temporal:
        run_temporal(retriever, args)
        return

    where = build_where(args)
    # Section is a free-text label; over-fetch and filter it in Python so users
    # can pass a prefix like "Item 1A" without knowing the full title.
    k = args.k * 6 if args.section else args.k
    hits = retriever.search(args.question, k=k, method=args.method, where=where)

    if args.section:
        needle = args.section.lower()
        hits = [h for h in hits if needle in h["metadata"].get("section", "").lower()]
    hits = hits[: args.k]

    filt = where or {}
    print(f'\nQ: "{args.question}"  ({args.method}, top {len(hits)} of '
          f'{retriever.collection.count()} chunks; filter={filt}'
          f'{" section~="+args.section if args.section else ""})\n')
    for rank, h in enumerate(hits, 1):
        meta, doc = h["metadata"], h["document"]
        print(f"[{rank}] {meta['ticker']} {meta['filing_type']} {meta['fiscal_period']}  |  "
              f"{meta['section']}  [{meta['chunk_kind']}]")
        snippet = textwrap.shorten(doc.replace("\n", " "), width=400, placeholder=" ...")
        print(textwrap.fill(snippet, width=100, initial_indent="    ", subsequent_indent="    "))
        print(f"    source: {meta['source_url']}\n")


if __name__ == "__main__":
    main()
