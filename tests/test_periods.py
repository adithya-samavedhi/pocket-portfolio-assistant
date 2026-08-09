"""Fiscal-period labelling — the bug that poisoned every quarterly citation.

Calendar-derived labels disagreed with the filing's own words for every filer
whose fiscal year doesn't start in January. These tests pin the fix: labels come
from the filing's inline XBRL, annual and quarterly periods stay distinguishable,
and a bare "Q2 2026" resolves to both readings rather than silently one.
"""
import json
from pathlib import Path

import pytest

from fetch import calendar_quarter, fiscal_focus, period_fields

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "raw" / "manifest.json"

# Two real-world shapes of the same dei tag. NVDA writes the value bare; MSFT
# wraps it in a styled <span>, which defeated the first parser and silently
# demoted those filings to a wrong calendar label.
NVDA_STYLE = (
    '<ix:nonNumeric contextRef="c-1" name="dei:DocumentFiscalYearFocus" id="f-27">2026'
    '</ix:nonNumeric><ix:nonNumeric contextRef="c-1" name="dei:DocumentFiscalPeriodFocus"'
    ' id="f-28">Q2</ix:nonNumeric>'
)
MSFT_STYLE = (
    '<ix:nonNumeric name="dei:DocumentFiscalYearFocus">'
    '<span style="color:#000000;font-weight:bold;">2026</span></ix:nonNumeric>'
    '<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus" contextRef="C_abc">Q3'
    '</ix:nonNumeric>'
)


def _write(tmp_path, html):
    p = tmp_path / "filing.htm"
    p.write_text(html)
    return p


class TestFiscalFocus:
    def test_reads_bare_value(self, tmp_path):
        assert fiscal_focus(_write(tmp_path, NVDA_STYLE)) == (2026, "Q2")

    def test_reads_value_wrapped_in_markup(self, tmp_path):
        """Regression: a <span> around the value must not hide it."""
        assert fiscal_focus(_write(tmp_path, MSFT_STYLE)) == (2026, "Q3")

    def test_numbers_inside_attributes_are_not_mistaken_for_the_value(self, tmp_path):
        """Only element text is the value — attributes carry digits too.

        These filings really do tag elements `id="f-27"`; an id like `f-2020`
        would otherwise be read as the fiscal year.
        """
        html = ('<ix:nonNumeric name="dei:DocumentFiscalYearFocus" id="f-2020">'
                '<span>2026</span></ix:nonNumeric>'
                '<ix:nonNumeric name="dei:DocumentFiscalPeriodFocus">Q2</ix:nonNumeric>')
        assert fiscal_focus(_write(tmp_path, html)) == (2026, "Q2")

    def test_missing_tags_report_nothing_rather_than_guessing(self, tmp_path):
        assert fiscal_focus(_write(tmp_path, "<html>no xbrl header</html>")) == (None, None)


class TestPeriodFields:
    def test_quarter_uses_the_companys_own_label(self, tmp_path):
        """NVDA's quarter ending 2025-07-27 is Q2 FY2026 — NOT calendar Q3 2025."""
        f = period_fields("10-Q", "2025-07-27", _write(tmp_path, NVDA_STYLE))
        assert f["fiscal_period"] == "Q2 FY2026"
        assert f["calendar_quarter"] == "Q3 2025"   # kept, but secondary
        assert f["period_type"] == "quarter"
        assert f["period_end"] == "2025-07-27"
        assert f["fiscal_source"] == "xbrl"

    def test_annual_is_labelled_and_typed_as_annual(self, tmp_path):
        html = NVDA_STYLE.replace(">Q2<", ">FY<")
        f = period_fields("10-K", "2026-01-25", _write(tmp_path, html))
        assert f["fiscal_period"] == "FY2026"
        assert f["period_type"] == "annual"

    def test_fallback_is_marked_so_it_cannot_pass_as_authoritative(self, tmp_path):
        f = period_fields("10-Q", "2025-07-27", _write(tmp_path, "<html/>"))
        assert f["fiscal_source"] == "calendar-fallback"
        assert f["fiscal_period"] == "Q3 2025"

    @pytest.mark.parametrize("date,expected", [
        ("2025-07-27", "Q3 2025"), ("2026-01-25", "Q1 2026"),
        ("2025-12-31", "Q4 2025"), ("2026-04-26", "Q2 2026"),
    ])
    def test_calendar_quarter(self, date, expected):
        assert calendar_quarter(date) == expected


@pytest.mark.skipif(not MANIFEST.exists(), reason="corpus not fetched")
class TestCorpusLabels:
    """Guard the real corpus, not just the parser."""

    @staticmethod
    def _entries():
        return json.loads(MANIFEST.read_text())

    def test_every_filing_has_an_authoritative_label(self):
        fallbacks = [e["file"] for e in self._entries()
                     if e.get("fiscal_source") != "xbrl"]
        assert not fallbacks, f"fell back to calendar labels: {fallbacks}"

    def test_known_offset_filers_are_labelled_the_companys_way(self):
        """The exact cases that were wrong before the fix."""
        by_file = {e["file"]: e["fiscal_period"] for e in self._entries()}
        expected = {
            "NVDA_10-Q_2025-07-27.htm": "Q2 FY2026",   # was "Q3 2025"
            "NVDA_10-Q_2026-04-26.htm": "Q1 FY2027",   # was "Q2 2026"
            "AAPL_10-Q_2025-12-27.htm": "Q1 FY2026",   # was "Q4 2025"
            "MSFT_10-Q_2026-03-31.htm": "Q3 FY2026",   # was "Q1 2026"
        }
        for f, want in expected.items():
            if f in by_file:
                assert by_file[f] == want, f"{f}: {by_file[f]} != {want}"

    def test_annual_and_quarterly_are_distinguishable(self):
        types = {e["period_type"] for e in self._entries()}
        assert types == {"annual", "quarter"}
