"""Company resolution — shared by the filings and market sides.

Comparative questions depend on spotting *every* company named, not just the
first: global top-k covered both named companies only 75% of the time, so the
model was asked to compare two companies while seeing passages from one.
"""
import config
from technicals import TICKERS, resolve_ticker


class TestMentionedTickers:
    def test_finds_both_companies_by_name(self):
        got = config.mentioned_tickers(
            "Compare how Microsoft and Amazon describe cloud growth")
        assert got == ["MSFT", "AMZN"]

    def test_mixes_symbols_and_names(self):
        assert config.mentioned_tickers("Is NVDA cheaper than Apple?") == ["NVDA", "AAPL"]

    def test_preserves_order_of_first_mention(self):
        assert config.mentioned_tickers("Alphabet, then NVIDIA") == ["GOOGL", "NVDA"]

    def test_deduplicates_repeated_mentions(self):
        assert config.mentioned_tickers("Apple's iPhone and Apple's Services") == ["AAPL"]

    def test_ignores_uncovered_companies(self):
        assert config.mentioned_tickers("Compare Tesla and Meta") == []

    def test_no_company_named(self):
        assert config.mentioned_tickers("What is a good P/E ratio?") == []


class TestResolveTicker:
    def test_single_company(self):
        assert resolve_ticker("What are NVIDIA's risk factors?") == "NVDA"

    def test_uncovered_returns_none(self):
        assert resolve_ticker("What is Tesla's revenue?") is None

    def test_technicals_reexports_the_shared_universe(self):
        """Both sides must resolve companies identically."""
        assert TICKERS is config.TICKERS
