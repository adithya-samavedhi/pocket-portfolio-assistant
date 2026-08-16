"""Whole-section expansion for analytical questions.

k=6 shows roughly 1.2% of a 10-K, but a moat or headwind argument is spread
across an entire item. Expansion fills the section back in — and, critically,
may only ever ADD: filling just the top-ranked sections silently dropped
passages chunk retrieval had already got right.
"""
from conftest import make_retriever


def _hit(i, ticker, period, section, text, idx, ftype="10-K"):
    return {"id": i, "document": text,
            "metadata": {"ticker": ticker, "fiscal_period": period, "section": section,
                         "filing_type": ftype, "chunk_index": idx, "chunk_kind": "text"}}


def _retriever(hits):
    return make_retriever(hits)


CORPUS = [
    _hit("a::0", "NVDA", "FY2026", "Item 1A: Risk Factors", "risk one", 0),
    _hit("a::1", "NVDA", "FY2026", "Item 1A: Risk Factors", "risk two", 1),
    _hit("a::2", "NVDA", "FY2026", "Item 1A: Risk Factors", "risk three", 2),
    _hit("a::3", "NVDA", "FY2026", "Item 7: MD&A", "mdna one", 3),
    _hit("b::0", "AAPL", "FY2025", "Item 1A: Risk Factors", "apple risk", 0),
]


class TestSectionChunks:
    def test_returns_the_whole_section_in_document_order(self):
        got = _retriever(CORPUS).section_chunks("NVDA", "FY2026", "Item 1A: Risk Factors")
        assert [h["document"] for h in got] == ["risk one", "risk two", "risk three"]

    def test_scoped_to_one_company_and_period(self):
        got = _retriever(CORPUS).section_chunks("AAPL", "FY2025", "Item 1A: Risk Factors")
        assert [h["id"] for h in got] == ["b::0"]


class TestExpandSections:
    def test_fills_in_the_rest_of_the_section(self):
        r = _retriever(CORPUS)
        out = r.expand_sections([CORPUS[1]], budget_chars=10_000)
        assert {h["id"] for h in out} >= {"a::0", "a::1", "a::2"}

    def test_never_drops_a_retrieved_passage(self):
        """Regression: expansion must be additive, never a replacement."""
        r = _retriever(CORPUS)
        seed = [CORPUS[3], CORPUS[4]]            # MD&A + a different company
        out = r.expand_sections(seed, budget_chars=10_000)
        for h in seed:
            assert h["id"] in {x["id"] for x in out}, f"{h['id']} was dropped"

    def test_respects_the_char_budget(self):
        r = _retriever(CORPUS)
        out = r.expand_sections([CORPUS[0]], budget_chars=len(CORPUS[0]["document"]))
        assert sum(len(h["document"]) for h in out) <= len(CORPUS[0]["document"])

    def test_no_duplicates(self):
        r = _retriever(CORPUS)
        out = r.expand_sections([CORPUS[0], CORPUS[1]], budget_chars=10_000)
        ids = [h["id"] for h in out]
        assert len(ids) == len(set(ids))
