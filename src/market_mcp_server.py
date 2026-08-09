"""MCP server exposing market data as tools (Phase 2, Milestone 2).

A second data source behind its own MCP boundary, alongside the filings server.
Returns structured summaries (not raw vendor JSON); responses are disk-cached.

Requires Python >=3.10.  Run:  python src/market_mcp_server.py   (stdio)
"""
from mcp.server import MCPServer

import marketdata

mcp = MCPServer("market-data")


@mcp.tool()
def get_quote(ticker: str) -> dict:
    """Current price, day change, and currency for a stock ticker (e.g. "AAPL")."""
    return marketdata.get_quote(ticker)


@mcp.tool()
def get_valuation(ticker: str) -> dict:
    """Valuation snapshot: market cap, trailing/forward P/E, P/S, 52-week range."""
    return marketdata.get_valuation(ticker)


@mcp.tool()
def get_price_history(ticker: str, window: str = "1y") -> dict:
    """Close-price history (downsampled) with start/end/high/low and % change.

    window: '1mo', '3mo', '6mo', '1y', '2y', '5y', etc.
    """
    return marketdata.get_price_history(ticker, window)


@mcp.tool()
def get_technicals(ticker: str) -> dict:
    """Trend/positioning metrics: 1mo-1y momentum, 50- and 200-day moving
    averages, position in the 52-week range, max drawdown, realized volatility.

    Use this for "how is it trending" or "where is it versus its own history".
    """
    return marketdata.get_technicals(ticker)


if __name__ == "__main__":
    mcp.run()
