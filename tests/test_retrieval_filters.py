"""Period filtering and temporal fan-out.

Two behaviours worth pinning: a bare "Q2 2026" must not silently resolve to one
reading, and a 10-K's fiscal year must not appear alongside the quarters it
contains (which invites a year-vs-quarter jump to be read as a trend).
"""
from retrieval import Retriever
from tools import _period_clause, _where


def _alts(clause):
    """Flatten a period clause into the set of (field, value) it will match."""
    branches = clause.get("$or", [clause])
    return {(k, v) for b in branches for k, v in b.items()}


class TestPeriodClause:
    def test_unambiguous_fiscal_label_matches_itself(self):
        assert ("fiscal_period", "Q2 FY2026") in _alts(_period_clause("Q2 FY2026"))

    def test_bare_quarter_matches_both_readings(self):
        """"Q2 2026" may mean fiscal Q2 FY2026 or calendar Q2 2026 — try both."""
        alts = _alts(_period_clause("Q2 2026"))
        assert ("fiscal_period", "Q2 FY2026") in alts
        assert ("calendar_quarter", "Q2 2026") in alts

    def test_fiscal_year_is_not_expanded_into_a_quarter(self):
        alts = _alts(_period_clause("FY2025"))
        assert alts == {("fiscal_period", "FY2025"), ("calendar_quarter", "FY2025")}

    def test_case_and_padding_tolerated(self):
        assert ("fiscal_period", "Q3 FY2026") in _alts(_period_clause("  q3 2026 "))


class TestWhereClause:
    def test_combines_filters(self):
        w = _where("NVDA", None, "10-Q", None)
        assert w == {"$and": [{"ticker": "NVDA"}, {"filing_type": "10-Q"}]}

    def test_no_filters_means_no_clause(self):
        assert _where(None, None, None, None) is None

    def test_period_type_is_filterable(self):
        w = _where("NVDA", None, None, "annual")
        assert {"period_type": "annual"} in w["$and"]


class TestPeriodsFor:
    """`periods_for` reads only self.metas, so drive it with a fake corpus."""

    @staticmethod
    def _retriever():
        r = Retriever.__new__(Retriever)
        r.metas = [
            {"ticker": "NVDA", "fiscal_period": "Q2 FY2026", "period_type": "quarter",
             "period_end": "2025-07-27", "filing_type": "10-Q"},
            {"ticker": "NVDA", "fiscal_period": "Q3 FY2026", "period_type": "quarter",
             "period_end": "2025-10-26", "filing_type": "10-Q"},
            {"ticker": "NVDA", "fiscal_period": "FY2026", "period_type": "annual",
             "period_end": "2026-01-25", "filing_type": "10-K"},
            {"ticker": "AAPL", "fiscal_period": "Q1 FY2026", "period_type": "quarter",
             "period_end": "2025-12-27", "filing_type": "10-Q"},
        ]
        return r

    def test_annual_excluded_by_default(self):
        """A fiscal year contains its own quarters — never fan out across both."""
        periods = [p for p, _ in self._retriever().periods_for("NVDA")]
        assert periods == ["Q2 FY2026", "Q3 FY2026"]
        assert "FY2026" not in periods

    def test_annual_reachable_when_asked_for(self):
        periods = [p for p, _ in self._retriever().periods_for("NVDA", period_type="annual")]
        assert periods == ["FY2026"]

    def test_ordered_chronologically_by_calendar_anchor(self):
        """Fiscal labels don't sort; period_end does."""
        got = self._retriever().periods_for("NVDA", period_type=None)
        assert [p for p, _ in got] == ["Q2 FY2026", "Q3 FY2026", "FY2026"]

    def test_scoped_to_one_ticker(self):
        assert [p for p, _ in self._retriever().periods_for("AAPL")] == ["Q1 FY2026"]
