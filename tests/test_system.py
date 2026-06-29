import unittest
import os
import sys
from pathlib import Path
from typing import List, Dict

# Add project root to sys path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings
from utils.logging import logger

from agents.scanner_agent import ScannerAgent

class TestSystemConfig(unittest.TestCase):
    def test_settings_load(self):
        """Test that settings defaults are loaded correctly."""
        self.assertIsNotNone(Settings.GEMINI_MODEL)
        self.assertIsNotNone(Settings.EMBEDDING_MODEL)
        self.assertIsNotNone(Settings.LOG_LEVEL)

    def test_api_key_validation(self):
        """Test that settings correctly validates presence or absence of API key."""
        original_key = Settings.GEMINI_API_KEY
        
        # Test empty key
        Settings.set_api_key("")
        self.assertFalse(Settings.validate())
        
        # Test non-empty key
        Settings.set_api_key("AIzaSyTestKey")
        self.assertTrue(Settings.validate())
        
        # Restore key
        Settings.set_api_key(original_key)

    def test_logger_creation(self):
        """Test that logging behaves properly and doesn't crash."""
        logger.info("Test log statement to check logger initialization.")
        self.assertTrue(Settings.LOG_FILE.exists())

class TestScannerAgent(unittest.TestCase):
    def setUp(self):
        self.scanner = ScannerAgent()

    def test_clean_text(self):
        """Test the pre-processing and cleaning logic of ScannerAgent."""
        raw_text = "   Breaking!  COVID-19 vaccine @#$% details revealed...  "
        expected = "breaking! covid-19 vaccine details revealed..."
        
        cleaned = self.scanner.clean_text(raw_text)
        self.assertEqual(cleaned, expected)

    def test_empty_input_validation(self):
        """Test that empty or blank input raises a ValueError."""
        with self.assertRaises(ValueError):
            self.scanner.clean_text("")
            
        with self.assertRaises(ValueError):
            self.scanner.clean_text("   ")

from agents.researcher_agent import ResearcherAgent, SearchProvider

class MockSearchProvider(SearchProvider):
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        return [{
            "source": "MockNews",
            "url": "https://mocknews.com/article",
            "snippet": f"Mock result for query: {query}"
        }]

class TestResearcherAgent(unittest.TestCase):
    def test_pluggable_search_provider(self):
        """Test that ResearcherAgent uses the configured pluggable search provider correctly."""
        mock_provider = MockSearchProvider()
        researcher = ResearcherAgent(provider=mock_provider)
        
        results = researcher.search_evidence("water freezing point", max_results=1)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["source"], "MockNews")
        self.assertEqual(results[0]["url"], "https://mocknews.com/article")
        self.assertIn("water freezing point", results[0]["snippet"])

from agents.fact_checker import FactCheckerAgent

class TestFactCheckerAgent(unittest.TestCase):
    def setUp(self):
        self.checker = FactCheckerAgent()

    def test_verdict_aggregation(self):
        """Test programmatic claims verdict aggregation logic in FactCheckerAgent."""
        claims = [
            {"id": "c1", "assertion": "Water boils at 100C"},
            {"id": "c2", "assertion": "The moon is made of cheese"}
        ]
        
        # Mock evidence synthesis
        evidence_by_claim = {
            "c1": {"synthesized_evidence": "Evidence says true", "sources": [{"source": "Science"}]},
            "c2": {"synthesized_evidence": "Evidence says false", "sources": [{"source": "NASA"}]}
        }
        
        # We temporarily mock verify_claim to avoid querying Gemini during unit tests
        original_verify = self.checker.verify_claim
        
        try:
            # Setup mock returns
            def mock_verify_claim(claim, evidence):
                if "cheese" in claim:
                    return {"classification": "False", "confidence": 0.9, "reason": "Moon is rocky"}
                return {"classification": "True", "confidence": 0.95, "reason": "Confirmed boiling point"}
                
            self.checker.verify_claim = mock_verify_claim
            
            report = self.checker.verify_claims(claims, evidence_by_claim)
            
            # Check overall verdict rule: since c2 is False, overall should be False
            self.assertEqual(report["overall_verdict"], "False")
            self.assertEqual(len(report["claim_verdicts"]), 2)
            self.assertEqual(report["claim_verdicts"][0]["verdict"], "True")
            self.assertEqual(report["claim_verdicts"][1]["verdict"], "False")
            
        finally:
            # Restore original method
            self.checker.verify_claim = original_verify

if __name__ == "__main__":
    unittest.main()
