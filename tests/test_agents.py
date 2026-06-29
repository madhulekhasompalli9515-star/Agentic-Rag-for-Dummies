import pytest
import sys
from pathlib import Path
from typing import List, Dict

# Ensure project root is in the path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from agents.scanner_agent import ScannerAgent
from agents.researcher_agent import ResearcherAgent, SearchProvider
from agents.fact_checker import FactCheckerAgent

# =====================================================================
# PYTEST FIXTURES
# =====================================================================
@pytest.fixture
def scanner():
    """Provides a ScannerAgent instance."""
    return ScannerAgent()

@pytest.fixture
def checker():
    """Provides a FactCheckerAgent instance."""
    return FactCheckerAgent()

# =====================================================================
# SCANNER AGENT TESTS
# =====================================================================
def test_scanner_clean_text(scanner):
    """Verifies that input text is preprocessed, lowercased, and sanitized."""
    raw_text = "   BREAKING!!   This claim has @#$% special chars   and extra whitespace...   "
    expected = "breaking!! this claim has special chars and extra whitespace..."
    
    cleaned = scanner.clean_text(raw_text)
    assert cleaned == expected

def test_scanner_empty_input_validation(scanner):
    """Ensures ScannerAgent throws ValueError on empty or blank inputs."""
    with pytest.raises(ValueError, match="Input news text cannot be empty"):
        scanner.clean_text("")
        
    with pytest.raises(ValueError, match="Input news text cannot be empty"):
        scanner.clean_text("     ")

# =====================================================================
# RESEARCHER AGENT TESTS
# =====================================================================
class DummySearchProvider(SearchProvider):
    """Mock search provider for testing pluggable interface."""
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        return [{
            "source": "FactArchive",
            "url": "https://factarchive.org/claim-check",
            "snippet": f"Factcheck search result for: {query}"
        }]

def test_researcher_custom_search_provider():
    """Verifies that the ResearcherAgent utilizes its pluggable search provider correctly."""
    mock_provider = DummySearchProvider()
    researcher = ResearcherAgent(provider=mock_provider)
    
    results = researcher.search_evidence("boiling point of water", max_results=1)
    
    assert len(results) == 1
    assert results[0]["source"] == "FactArchive"
    assert results[0]["url"] == "https://factarchive.org/claim-check"
    assert "boiling point of water" in results[0]["snippet"]

# =====================================================================
# FACT CHECKER AGENT TESTS
# =====================================================================
def test_fact_checker_verdict_aggregation(checker, monkeypatch):
    """Verifies programmatic verdict aggregation in FactCheckerAgent using mocked LLM outputs."""
    claims = [
        {"id": "c1", "assertion": "Assertion 1"},
        {"id": "c2", "assertion": "Assertion 2"},
        {"id": "c3", "assertion": "Assertion 3"}
    ]
    
    evidence_by_claim = {
        "c1": {"synthesized_evidence": "Evidence 1", "sources": []},
        "c2": {"synthesized_evidence": "Evidence 2", "sources": []},
        "c3": {"synthesized_evidence": "Evidence 3", "sources": []}
    }
    
    # Mock individual claim verification function to bypass Gemini API calls
    def mock_verify_claim(assertion, evidence):
        if "Assertion 1" in assertion:
            return {"classification": "True", "confidence": 0.95, "reason": "Verified True"}
        elif "Assertion 2" in assertion:
            return {"classification": "Misleading", "confidence": 0.7, "reason": "Verified Misleading"}
        else:
            return {"classification": "Insufficient Evidence", "confidence": 0.5, "reason": "No evidence"}
            
    monkeypatch.setattr(checker, "verify_claim", mock_verify_claim)
    
    # Call verify_claims to test logical routing
    report = checker.verify_claims(claims, evidence_by_claim)
    
    # Rules check: Since no "False" verdicts exist, but "Misleading" is present,
    # the overall verdict must compile to "Misleading".
    assert report["overall_verdict"] == "Misleading"
    assert len(report["claim_verdicts"]) == 3
    assert report["claim_verdicts"][0]["verdict"] == "True"
    assert report["claim_verdicts"][1]["verdict"] == "Misleading"
    assert report["claim_verdicts"][2]["verdict"] == "Insufficient Evidence"
