"""The sub-agent hand-off contract.

Both behaviours here were bugs found in live runs: answers citing [7] under a
five-source list, and sub-agents reporting "no evidence" with high confidence.
"""
from contracts import SUMMARY_CHAR_BUDGET, AgentAnswer, Citation, renumber

FILING = dict(ticker="NVDA", filing_type="10-Q", fiscal_period="Q2 FY2026",
              section="Item 2", source_url="https://sec.gov/x")


class TestRenumber:
    def test_markers_follow_the_deduped_list(self):
        """Regression: an answer must never cite [7] under a 5-source list."""
        assert renumber("high [1], [3] grew [7].", {1: 1, 3: 3, 7: 4}) == \
            "high [1], [3] grew [4]."

    def test_grouped_markers_are_split(self):
        assert renumber("Both [1, 3] agree.", {1: 1, 3: 2}) == "Both [1], [2] agree."

    def test_duplicate_sources_collapse_to_one_number(self):
        assert renumber("See [2] and [5].", {2: 1, 5: 1}) == "See [1] and [1]."

    def test_unmapped_marker_is_dropped_not_left_dangling(self):
        assert renumber("Unmapped [9] gone.", {1: 1}) == "Unmapped gone."

    def test_text_without_markers_is_untouched(self):
        assert renumber("No markers here.", {}) == "No markers here."


class TestAgentAnswer:
    def test_insufficient_evidence_forces_low_confidence(self):
        """"I found nothing, and I'm sure of it" misleads the orchestrator."""
        a = AgentAnswer(agent="fundamentals", answer="nothing found",
                        confidence="high", insufficient_evidence=True)
        assert a.confidence == "low"

    def test_insufficient_evidence_drops_citations(self):
        a = AgentAnswer(agent="fundamentals", answer="nothing found",
                        citations=[Citation(**FILING)], insufficient_evidence=True)
        assert a.citations == []

    def test_normal_answer_is_left_alone(self):
        a = AgentAnswer(agent="fundamentals", answer="found it",
                        citations=[Citation(**FILING)], confidence="high")
        assert a.confidence == "high" and len(a.citations) == 1

    def test_budget_tracks_answer_and_citations(self):
        small = AgentAnswer(agent="x", answer="short")
        assert small.within_budget()
        big = AgentAnswer(agent="x", answer="y" * (SUMMARY_CHAR_BUDGET + 1))
        assert not big.within_budget()

    def test_citation_label_names_the_period_the_company_uses(self):
        assert Citation(**FILING).label() == "NVDA 10-Q Q2 FY2026 — Item 2"
