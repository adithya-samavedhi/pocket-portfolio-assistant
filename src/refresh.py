"""Keep the corpus current: fetch newly filed documents, then index them.

    python src/refresh.py                     # check for new filings, index them
    python src/refresh.py --status            # just report how stale things are
    python src/refresh.py --add TSLA META     # start tracking more companies
    python src/refresh.py --quarters 20       # deepen history (5 years of 10-Qs)

Tracking N companies over several years is a corpus-management problem, not a
retrieval one: you cannot re-upload documents every time you ask a trend
question. This is the maintenance loop for that corpus.

Existing filings are never re-downloaded (EDGAR rate-limits, and it's rude) and
never re-embedded, so a refresh costs only what is genuinely new. The manifest
is MERGED rather than replaced, so deepening history or adding a company can't
silently drop what you already had.
"""
import argparse
import json
import subprocess
import sys
from datetime import date, datetime
from pathlib import Path

import config
import fetch

MANIFEST = config.RAW / "manifest.json"


def load():
    return json.loads(MANIFEST.read_text()) if MANIFEST.exists() else []


def tracked(entries):
    return sorted({e["ticker"] for e in entries})


def status(entries):
    if not entries:
        print("  corpus is empty — run: python src/fetch.py && python src/ingest.py --all")
        return
    newest = max(e["filing_date"] for e in entries)
    age = (date.today() - datetime.fromisoformat(newest).date()).days
    print(f"  companies : {len(tracked(entries))} — {', '.join(tracked(entries))}")
    print(f"  filings   : {len(entries)}")
    print(f"  newest    : {newest}  ({age} days old)")
    by_ticker = {}
    for e in entries:
        by_ticker.setdefault(e["ticker"], []).append(e)
    thin = [t for t, v in by_ticker.items() if len(v) < 4]
    if thin:
        print(f"  thin cover: {', '.join(thin)} (fewer than 4 filings)")
    if age > 45:
        print("  -> a quarter has probably been filed since; refresh")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", action="store_true", help="report freshness and exit")
    ap.add_argument("--add", nargs="+", metavar="TICKER", default=[],
                    help="companies to start tracking")
    ap.add_argument("--quarters", type=int, default=None,
                    help="10-Qs to keep per company (default: keep current depth)")
    ap.add_argument("--no-ingest", action="store_true", help="fetch only, don't index")
    args = ap.parse_args()

    entries = load()
    if args.status:
        status(entries)
        return 0

    print("Before:")
    status(entries)

    tickers = sorted(set(tracked(entries)) | {t.upper() for t in args.add})
    if not tickers:
        tickers = fetch.DEFAULT_TICKERS
    # Keep whatever depth the corpus already has unless told otherwise.
    depth = args.quarters or max(
        [sum(1 for e in entries if e["ticker"] == t and e["filing_type"] == "10-Q")
         for t in tracked(entries)] or [3])

    print(f"\nChecking EDGAR for {len(tickers)} companies ({depth} quarters each)...")
    config.CACHE.mkdir(parents=True, exist_ok=True)
    cik_map = fetch.ticker_to_cik()

    fetched = []
    for t in tickers:
        if t not in cik_map:
            print(f"  {t:<6} SKIPPED — not in EDGAR's ticker map")
            continue
        fetched.extend(fetch.fetch_ticker(t, cik_map[t], depth))

    # Merge on filename: new filings are added, existing ones keep their entry.
    by_file = {e["file"]: e for e in entries}
    added = [e for e in fetched if e["file"] not in by_file]
    for e in fetched:
        by_file[e["file"]] = e
    merged = sorted(by_file.values(), key=lambda e: (e["ticker"], e["report_date"]))
    MANIFEST.write_text(json.dumps(merged, indent=2))

    print(f"\n{len(added)} new filing(s); manifest now {len(merged)}.")
    if added:
        for e in added:
            print(f"  + {e['ticker']:<6} {e['filing_type']:<5} {e['fiscal_period']}")

    if args.no_ingest:
        print("\nSkipping indexing (--no-ingest). Run: python src/ingest.py --all")
        return 0

    print("\nIndexing (already-indexed filings are skipped)...")
    rc = subprocess.call([sys.executable, str(Path(__file__).parent / "ingest.py"), "--all"])
    if rc != 0:
        return rc

    print("\nAfter:")
    status(load())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
