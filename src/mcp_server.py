"""MCP server exposing filings retrieval as a tool (Milestone 6).

Wraps the same `search_filings` tool boundary the agent uses, so any MCP client
(Claude Desktop, the Phase 2 orchestrator, etc.) can query the corpus. Returns
structured summaries with source metadata, never a raw dump.

Requires Python >=3.10 (the mcp SDK's floor).
Run:  python src/mcp_server.py       (stdio transport)
"""
from typing import List, Dict, Optional

from mcp.server import MCPServer

import tools

mcp = MCPServer("filings")


@mcp.tool()
def search_filings(query: str, ticker: Optional[str] = None,
                   section: Optional[str] = None, period: Optional[str] = None,
                   filing_type: Optional[str] = None, k: int = 6,
                   temporal: bool = False) -> List[Dict]:
    """Search SEC 10-K/10-Q filings and return grounded passages with metadata.

    query       natural-language question or keywords
    ticker      restrict to one company, e.g. "NVDA"
    section     SEC item label substring, e.g. "Item 1A"
    period      fiscal period, e.g. "Q2 2026" or "FY2025"
    filing_type "10-K" or "10-Q"
    k           number of passages (per period when temporal=True)
    temporal    fan out across a ticker's periods (requires ticker) for
                change-over-time questions, so every period is represented
    """
    return tools.search_filings(
        query, ticker=ticker, section=section, period=period,
        filing_type=filing_type, k=k, temporal=temporal)


if __name__ == "__main__":
    mcp.run()
