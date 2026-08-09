"""Market-data tool boundary (Phase 2, Milestone 2).

Three tools with clean boundaries — `get_quote`, `get_valuation`,
`get_price_history` — returning structured summaries, not raw vendor JSON.

Provider: yfinance by default (keyless, and its free history beats Finnhub's).
If FINNHUB_API_KEY is set, quotes and valuation come from Finnhub instead (the
official, more reliable path preferred for scheduled use). Every response is
cached to disk with a TTL; network calls retry with exponential backoff.
"""
import json
import os
import sys
import time
from datetime import date

import requests

import config

MARKET_CACHE = config.CACHE / "market"
QUOTE_TTL = 15 * 60          # quotes move intraday
VALUATION_TTL = 6 * 3600
HISTORY_TTL = 12 * 3600
TECHNICALS_TTL = 12 * 3600
_MAX_POINTS = 30             # downsample history to keep the summary bounded

# Trading-day offsets used for momentum and moving averages.
_TD = {"1mo": 21, "3mo": 63, "6mo": 126, "1y": 252}


def _log(msg: str):
    print(f"[market] {msg}", file=sys.stderr)


def _cached(key: str, ttl: int, producer):
    MARKET_CACHE.mkdir(parents=True, exist_ok=True)
    path = MARKET_CACHE / f"{key}.json"
    if path.exists() and time.time() - path.stat().st_mtime < ttl:
        _log(f"cache HIT  {key}")
        return json.loads(path.read_text())
    _log(f"cache MISS {key}")
    data = producer()
    path.write_text(json.dumps(data))
    return data


def _retry(fn, tries: int = 3):
    for i in range(tries):
        try:
            return fn()
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(0.5 * 2 ** i)   # exponential backoff


# --- Finnhub (used only when a key is present) ---
def _finnhub_key():
    return os.environ.get("FINNHUB_API_KEY")


def _finnhub_get(path, **params):
    params["token"] = _finnhub_key()
    r = requests.get(f"https://finnhub.io/api/v1/{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def _finnhub_quote(ticker):
    q = _finnhub_get("quote", symbol=ticker)
    return {"ticker": ticker, "price": q.get("c"), "currency": "USD",
            "change": q.get("d"), "change_pct": q.get("dp"),
            "as_of": date.today().isoformat(), "source": "finnhub"}


def _finnhub_valuation(ticker):
    m = _finnhub_get("stock/metric", symbol=ticker, metric="all").get("metric", {})
    return {"ticker": ticker, "price": None, "currency": "USD",
            "market_cap": m.get("marketCapitalization"),
            "pe_trailing": m.get("peTTM"), "pe_forward": m.get("forwardPE"),
            "price_to_sales": m.get("psTTM"),
            "week52_high": m.get("52WeekHigh"), "week52_low": m.get("52WeekLow"),
            "as_of": date.today().isoformat(), "source": "finnhub"}


# --- yfinance (default) ---
def _yf_quote(ticker):
    import yfinance as yf
    fi = yf.Ticker(ticker).fast_info
    price, prev = float(fi.last_price), float(fi.previous_close)
    return {"ticker": ticker, "price": round(price, 2), "currency": fi.currency,
            "change": round(price - prev, 2),
            "change_pct": round((price - prev) / prev * 100, 2),
            "as_of": date.today().isoformat(), "source": "yfinance"}


def _yf_valuation(ticker):
    import yfinance as yf
    t = yf.Ticker(ticker)
    info, fi = t.info, t.fast_info
    return {"ticker": ticker, "price": round(float(fi.last_price), 2),
            "currency": fi.currency, "market_cap": info.get("marketCap"),
            "pe_trailing": info.get("trailingPE"), "pe_forward": info.get("forwardPE"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "week52_high": info.get("fiftyTwoWeekHigh"),
            "week52_low": info.get("fiftyTwoWeekLow"),
            "as_of": date.today().isoformat(), "source": "yfinance"}


def _yf_history(ticker, window):
    import yfinance as yf
    h = yf.Ticker(ticker).history(period=window)
    closes = h["Close"].dropna()
    base = {"ticker": ticker, "window": window, "source": "yfinance"}
    if closes.empty:
        return {**base, "points": [], "note": "no data"}
    step = max(1, len(closes) // _MAX_POINTS)
    points = [{"date": idx.date().isoformat(), "close": round(float(v), 2)}
              for idx, v in list(closes.items())[::step]]
    start, end = float(closes.iloc[0]), float(closes.iloc[-1])
    return {**base,
            "start_date": closes.index[0].date().isoformat(),
            "end_date": closes.index[-1].date().isoformat(),
            "start_close": round(start, 2), "end_close": round(end, 2),
            "high": round(float(closes.max()), 2), "low": round(float(closes.min()), 2),
            "pct_change": round((end - start) / start * 100, 2),
            "points": points}


def _yf_technicals(ticker):
    """Derived trend/positioning metrics computed from the full daily series.

    The full series stays in here: only the derived numbers cross the tool
    boundary. Two years are pulled so the 200-day average and the drawdown have
    real history behind them, while the 52-week figures use the last 252 days.
    """
    import yfinance as yf
    closes = yf.Ticker(ticker).history(period="2y")["Close"].dropna()
    base = {"ticker": ticker, "source": "yfinance"}
    if len(closes) < 30:
        return {**base, "note": "not enough history"}

    last = float(closes.iloc[-1])
    out = {**base,
           "price": round(last, 2),
           "as_of": closes.index[-1].date().isoformat(),
           "history_days": len(closes)}

    # Momentum: % change over each lookback we have enough data for.
    out["returns_pct"] = {
        label: round((last / float(closes.iloc[-1 - n]) - 1) * 100, 2)
        for label, n in _TD.items() if len(closes) > n
    }

    # Moving averages and where price sits relative to them.
    for n in (50, 200):
        if len(closes) >= n:
            ma = float(closes.iloc[-n:].mean())
            out[f"ma{n}"] = round(ma, 2)
            out[f"pct_vs_ma{n}"] = round((last / ma - 1) * 100, 2)
    if "ma50" in out and "ma200" in out:
        # Golden/death cross state — the trend regime, not a trade signal.
        out["ma50_above_ma200"] = out["ma50"] > out["ma200"]

    # 52-week range and where in it the price sits (0 = at the low, 100 = high).
    yr = closes.iloc[-_TD["1y"]:] if len(closes) > _TD["1y"] else closes
    hi, lo = float(yr.max()), float(yr.min())
    out["week52_high"], out["week52_low"] = round(hi, 2), round(lo, 2)
    out["pct_from_52w_high"] = round((last / hi - 1) * 100, 2)
    if hi > lo:
        out["range_position_pct"] = round((last - lo) / (hi - lo) * 100, 1)

    # Worst peak-to-trough decline over the last year, and realized vol.
    out["max_drawdown_pct"] = round(float((yr / yr.cummax() - 1).min()) * 100, 2)
    out["volatility_annualized_pct"] = round(
        float(yr.pct_change().dropna().std()) * (252 ** 0.5) * 100, 1)

    # No keyless source of historical P/E, so "expensive vs. its own history"
    # is answerable on price/trend only. Stated explicitly so the sub-agent
    # cannot quietly imply a valuation-history comparison it never made.
    out["pe_history_available"] = False
    return out


# --- public tool boundary ---
def get_quote(ticker: str) -> dict:
    """Current price, day change, and currency for a ticker."""
    producer = _finnhub_quote if _finnhub_key() else _yf_quote
    return _cached(f"quote_{ticker}", QUOTE_TTL, lambda: _retry(lambda: producer(ticker)))


def get_valuation(ticker: str) -> dict:
    """Valuation snapshot: market cap, P/E (trailing/forward), P/S, 52-week range."""
    producer = _finnhub_valuation if _finnhub_key() else _yf_valuation
    return _cached(f"valuation_{ticker}", VALUATION_TTL, lambda: _retry(lambda: producer(ticker)))


def get_price_history(ticker: str, window: str = "1y") -> dict:
    """Downsampled close-price series plus start/end/high/low and % change.

    window: any yfinance period, e.g. '1mo', '3mo', '6mo', '1y', '2y', '5y'.
    History always uses yfinance (Finnhub's free tier no longer serves candles).
    """
    return _cached(f"history_{ticker}_{window}", HISTORY_TTL,
                   lambda: _retry(lambda: _yf_history(ticker, window)))


def get_technicals(ticker: str) -> dict:
    """Trend and positioning metrics: momentum, 50/200-day averages, 52-week
    position, max drawdown, realized volatility. Computed from the full daily
    series so the arithmetic is deterministic and the series never leaves here.
    """
    return _cached(f"technicals_{ticker}", TECHNICALS_TTL,
                   lambda: _retry(lambda: _yf_technicals(ticker)))
