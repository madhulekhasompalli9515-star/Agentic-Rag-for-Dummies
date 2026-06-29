from typing import Generator, Dict, Any
from agents.scanner_agent import ScannerAgent
from agents.researcher_agent import ResearcherAgent
from agents.fact_checker import FactCheckerAgent
from backend.rag_engine import RAGEngine
from config.settings import Settings
from utils.logging import logger

class Orchestrator:
    def __init__(self):
        """Initializes the orchestrator and all agents."""
        self.scanner = ScannerAgent()
        self.researcher = ResearcherAgent()
        self.fact_checker = FactCheckerAgent()
        self.rag_engine = RAGEngine()

    def run_pipeline(self, text: str) -> Generator[Dict[str, Any], None, None]:
        """
        Runs the complete Agentic RAG pipeline.
        Yields dictionaries indicating current status and payload so the frontend can update in real-time.
        """
        logger.info("[Orchestrator] Starting verification pipeline...")
        
        # Initialize Agent models dynamically (handles case where API key was updated in UI)
        try:
            self.scanner.init_model()
            self.researcher.init_model()
            self.fact_checker.init_model()
        except Exception as e:
            logger.error(f"[Orchestrator] Model initialization failed: {e}")
            yield {
                "status": "error",
                "message": f"Model Initialization Failed: {str(e)}. Please check your GEMINI_API_KEY.",
                "data": None
            }
            return

        # ----------------------------------------------------
        # Step 1: Scan text and extract assertions
        # ----------------------------------------------------
        yield {
            "status": "scanning",
            "message": "Scanner Agent: Extracting testable claims and checking sensationalism score...",
            "data": None
        }
        
        try:
            scan_results = self.scanner.scan_article(text)
            yield {
                "status": "scanning_complete",
                "message": f"Scanner Agent: Extracted {len(scan_results.get('claims', []))} claims.",
                "data": scan_results
            }
        except Exception as e:
            logger.error(f"[Orchestrator] Scanner agent error: {e}")
            yield {
                "status": "error",
                "message": f"Scanner Agent failed: {str(e)}",
                "data": None
            }
            return

        claims = scan_results.get("claims", [])
        evidence_by_claim = {}

        if not claims:
            logger.warning("[Orchestrator] Scanner Agent found 0 fact-checkable claims.")
            yield {
                "status": "completed",
                "message": "Orchestrator: No fact-checkable claims were identified in the source text.",
                "data": {
                    "scan_results": scan_results,
                    "evidence_by_claim": {},
                    "fact_check_results": {
                        "overall_verdict": "UNVERIFIED",
                        "overall_reasoning": "No testable factual assertions were extracted from the input text.",
                        "claim_verdicts": []
                    }
                }
            }
            return

        # ----------------------------------------------------
        # Step 2: Research claims (Web Search + Local RAG)
        # ----------------------------------------------------
        for index, claim in enumerate(claims):
            claim_id = claim.get("id", f"claim_{index}")
            assertion = claim.get("assertion", "")
            search_query = claim.get("search_query", assertion)
            
            yield {
                "status": "researching",
                "message": f"Researcher Agent: Researching Claim {index+1}/{len(claims)}: '{assertion[:60]}...'",
                "data": {"current": index + 1, "total": len(claims), "assertion": assertion}
            }
            
            # Perform RAG / Search retrieval
            web_evidence = self.researcher.search_evidence(search_query, max_results=Settings.MAX_SEARCH_RESULTS)
            local_evidence = self.rag_engine.query_kb(search_query, n_results=3)
            
            # Combine search and RAG sources
            combined_sources = web_evidence + local_evidence
            
            try:
                # Synthesize evidence
                synthesis = self.researcher.research_claim(
                    claim_assertion=assertion,
                    claim_context=claim.get("context", ""),
                    search_results=combined_sources
                )
                evidence_by_claim[claim_id] = synthesis
            except Exception as e:
                logger.error(f"[Orchestrator] Researcher agent error on claim '{assertion}': {e}")
                evidence_by_claim[claim_id] = {
                    "synthesized_evidence": f"Failed to synthesize evidence due to error: {str(e)}",
                    "key_findings": ["Error retrieving findings."],
                    "sources": combined_sources
                }

        # ----------------------------------------------------
        # Step 3: Fact-Check assertions against synthesized evidence
        # ----------------------------------------------------
        yield {
            "status": "fact_checking",
            "message": "Fact Checker Agent: Running logical claim evaluation against evidence...",
            "data": None
        }
        
        try:
            fact_check_results = self.fact_checker.verify_claims(claims, evidence_by_claim)
            yield {
                "status": "completed",
                "message": "Verification pipeline completed successfully!",
                "data": {
                    "scan_results": scan_results,
                    "evidence_by_claim": evidence_by_claim,
                    "fact_check_results": fact_check_results
                }
            }
        except Exception as e:
            logger.error(f"[Orchestrator] Fact Checker agent error: {e}")
            yield {
                "status": "error",
                "message": f"Fact Checker Agent failed: {str(e)}",
                "data": None
            }
