"""Fetch SEC filings (latest 10-K + recent 10-Qs) for a set of tickers.

Usage:
    python src/fetch.py                       # default 5-company Phase 1 set
    python src/fetch.py NVDA AAPL             # specific tickers
    python src/fetch.py --quarters 4          # how many recent 10-Qs each

Raw HTML is saved under data/raw/<TICKER>/ and never re-downloaded if present
(EDGAR rate limits, and it's rude). A combined manifest is written to
data/raw/manifest.json for the ingest step.
"""
import argparse
import json
import re
import time

import requests

import config

HEADERS = {"User-Agent": config.USER_AGENT}  # EDGAR requires an identifying UA

DEFAULT_TICKERS = ["GOOGL", "NVDA", "MSFT", "AAPL", "AMZN"]
TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"


def get(url, dest=None):
    """GET with the required UA header; polite pause; optional save to disk."""
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    time.sleep(0.4)
    if dest:
        dest.write_bytes(r.content)
    return r


def ticker_to_cik():
    cache = config.CACHE / "company_tickers.json"
    if not cache.exists():
        get(TICKER_MAP_URL, cache)
    data = json.loads(cache.read_text())
    return {row["ticker"]: int(row["cik_str"]) for row in data.values()}


def calendar_quarter(report_date):
    """The calendar quarter a period ended in — e.g. 'Q3 2025' for 2025-07-27.

    Kept as a secondary label only. It is what users mean when they say "Q2
    2026" in market terms, and it is the bridge to price data, but it is NOT
    what the company calls the period.
    """
    y, m, _ = (int(x) for x in report_date.split("-"))
    return f"Q{(m - 1) // 3 + 1} {y}"


# The company's own fiscal labelling, declared in the filing's inline XBRL:
#   dei:DocumentFiscalYearFocus   -> 2026
#   dei:DocumentFiscalPeriodFocus -> Q2 | Q3 | FY
# This is authoritative — it is the vocabulary the filing text itself uses —
# and it is why calendar-derived quarter labels were wrong for every filer
# whose fiscal year doesn't start in January (NVDA, AAPL, MSFT...).
_TAGS = re.compile(r"<[^>]*>")


def _focus_value(head, tag, pattern):
    """Pull a dei focus value, tolerating markup wrapped around it.

    Filers differ: NVDA writes `...FiscalYearFocus" id="f-27">2026</ix:nonNumeric>`
    while MSFT wraps the value in a styled `<span>`. So take a window after the
    tag name, strip any nested markup, and read the first matching token.
    """
    i = head.find(tag)
    if i == -1:
        return None
    # Step past the rest of the opening tag first — its own attributes carry
    # digits (`id="f-27"`) that would otherwise be read as the value — then
    # strip any nested markup wrapping the text.
    j = head.find(">", i)
    if j == -1:
        return None
    window = _TAGS.sub(" ", head[j + 1: j + 301])
    m = re.search(pattern, window)
    return m.group(1) if m else None


def fiscal_focus(path):
    """Return (fiscal_year, fiscal_quarter) from a filing's iXBRL, or (None, None).

    Reads only the head of the file: the dei header block sits near the top,
    and these documents run to several megabytes.
    """
    try:
        head = path.read_text(errors="ignore")[:400_000]
    except OSError:
        return None, None
    fy = _focus_value(head, "DocumentFiscalYearFocus", r"\b([12]\d{3})\b")
    fp = _focus_value(head, "DocumentFiscalPeriodFocus", r"\b(FY|Q[1-4])\b")
    return (int(fy) if fy else None), fp


def period_fields(form, report_date, path):
    """Build every period field for one filing, preferring the company's own labels.

    Falls back to a calendar label when a filing carries no iXBRL header (older
    or non-standard filings). `fiscal_source` records which path was taken so a
    silently-wrong label can never masquerade as an authoritative one.
    """
    fy, fq = fiscal_focus(path)
    cal = calendar_quarter(report_date)
    annual = form == "10-K"

    if fy and fq:
        label = f"FY{fy}" if fq == "FY" else f"{fq} FY{fy}"
        source = "xbrl"
    else:
        # No declared focus — say so rather than pretend.
        y = int(report_date[:4])
        label, fy, fq = (f"FY{y}", y, "FY") if annual else (cal, y, None)
        source = "calendar-fallback"

    return {
        "fiscal_period": label,                       # primary: "Q2 FY2026" / "FY2026"
        "fiscal_year": fy,
        "fiscal_quarter": fq or "",                   # "Q2" | "FY" | ""
        "period_type": "annual" if annual else "quarter",
        "period_end": report_date,                    # calendar anchor + market-data join key
        "calendar_quarter": cal,                      # "Q3 2025"
        "fiscal_source": source,
    }


def select_filings(cik, quarters):
    """Return the latest 10-K plus the `quarters` most recent 10-Qs."""
    cache = config.CACHE / f"CIK{cik:010d}_submissions.json"
    if not cache.exists():
        get(SUBMISSIONS_URL.format(cik=cik), cache)
    recent = json.loads(cache.read_text())["filings"]["recent"]
    keys = ["accessionNumber", "form", "filingDate", "reportDate", "primaryDocument"]
    rows = [dict(zip(keys, vals)) for vals in zip(*[recent[k] for k in keys])]

    tenk = next((r for r in rows if r["form"] == "10-K"), None)
    tenqs = [r for r in rows if r["form"] == "10-Q"][:quarters]
    return ([tenk] if tenk else []) + tenqs


def fetch_ticker(ticker, cik, quarters):
    out_dir = config.RAW / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for f in select_filings(cik, quarters):
        acc_nodash = f["accessionNumber"].replace("-", "")
        local = f"{ticker}_{f['form']}_{f['reportDate']}.htm"
        dest = out_dir / local
        url = ARCHIVE_URL.format(cik=cik, acc=acc_nodash, doc=f["primaryDocument"])
        if dest.exists():
            status = "cached"
        else:
            try:
                get(url, dest)
                status = f"{dest.stat().st_size:,} bytes"
            except requests.RequestException as e:
                print(f"  {ticker:<5} {f['form']:<5} {f['reportDate']}  FAILED: {e}")
                continue
        entry = {
            "file": local,
            "ticker": ticker,
            "cik": f"{cik:010d}",
            "filing_type": f["form"],
            **period_fields(f["form"], f["reportDate"], dest),
            "report_date": f["reportDate"],
            "filing_date": f["filingDate"],
            "source_url": url,
        }
        print(f"  {ticker:<5} {f['form']:<5} {f['reportDate']}  "
              f"{entry['fiscal_period']:<12} {status}")
        entries.append(entry)
    return entries


def relabel(manifest_path):
    """Rebuild period fields for an existing manifest from local files only.

    Re-deriving labels must not mean re-downloading 20 filings from EDGAR.
    """
    entries = json.loads(manifest_path.read_text())
    changed = 0
    for e in entries:
        path = config.RAW / e["ticker"] / e["file"]
        if not path.exists():
            print(f"  {e['file']}: MISSING locally — left unchanged")
            continue
        old = e.get("fiscal_period")
        e.update(period_fields(e["filing_type"], e["report_date"], path))
        mark = "" if old == e["fiscal_period"] else f"   was {old}"
        changed += old != e["fiscal_period"]
        print(f"  {e['ticker']:<6} {e['filing_type']:<5} {e['report_date']}  "
              f"-> {e['fiscal_period']:<12} [{e['fiscal_source']}]{mark}")
    manifest_path.write_text(json.dumps(entries, indent=2))
    print(f"\nRelabelled {len(entries)} filings; {changed} label(s) changed.")
    return entries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tickers", nargs="*", default=DEFAULT_TICKERS)
    ap.add_argument("--quarters", type=int, default=3, help="recent 10-Qs per ticker")
    ap.add_argument("--relabel", action="store_true",
                    help="re-derive period fields for the existing manifest from "
                         "local files; downloads nothing")
    args = ap.parse_args()
    tickers = args.tickers or DEFAULT_TICKERS

    if args.relabel:
        relabel(config.RAW / "manifest.json")
        print("Re-ingest to apply: python src/ingest.py --all --reset")
        return

    config.CACHE.mkdir(parents=True, exist_ok=True)
    cik_map = ticker_to_cik()

    manifest = []
    for t in tickers:
        if t not in cik_map:
            print(f"  {t:<5} SKIPPED — not found in EDGAR ticker map")
            continue
        manifest.extend(fetch_ticker(t, cik_map[t], args.quarters))

    manifest_path = config.RAW / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {manifest_path} with {len(manifest)} filings "
          f"across {len(tickers)} tickers.")


if __name__ == "__main__":
    main()
