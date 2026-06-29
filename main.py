import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

# Ensure the project root is in the path
project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.settings import Settings
from backend.orchestrator import Orchestrator
from utils.logging import logger

def print_banner():
    print("=" * 70)
    print("      [v]  VERITRUTH: AGENTIC RAG FAKE NEWS DETECTION SYSTEM  [v]")
    print("=" * 70)

def display_results(results: dict):
    """
    Renders verification results in a structured terminal layout.
    """
    scan = results.get("scan_results", {})
    fact_check = results.get("fact_check_results", {})
    evidence = results.get("evidence_by_claim", {})
    
    print("\n" + "=" * 50)
    print("                [v]  VERDICT REPORT  [v]")
    print("=" * 50)
    
    # 1. Overall Status
    overall_verdict = fact_check.get("overall_verdict", "UNVERIFIED").upper()
    print(f"\nOVERALL SYSTEM VERDICT :  [{overall_verdict}]")
    print(f"SENSATIONALISM SCORE   :  {scan.get('sensationalism_score', 0)}/10")
    
    # 2. General Reasoning
    print(f"\nHIGH-LEVEL SUMMARY REASONING:")
    print("-" * 35)
    print(fact_check.get("overall_reasoning", "No overall reasoning supplied."))
    
    # 3. Keywords & Entities
    keywords = scan.get("keywords", [])
    if keywords:
        print(f"\nEXTRACTED KEYWORDS: {', '.join(keywords)}")
        
    entities = scan.get("entities", {})
    if any(entities.values() if isinstance(entities, dict) else []):
        print("\nIDENTIFIED NAMED ENTITIES:")
        for category, items in entities.items():
            if items:
                print(f"  - {category.capitalize()}: {', '.join(items)}")
                
    biases = scan.get("biases_and_fallacies", [])
    if biases:
        print(f"\nDETECTED BIASES & FALLACIES: {', '.join(biases)}")
        
    # 4. Claim breakdown
    print("\n" + "=" * 50)
    print("              DETAILED CLAIM VERDICTS")
    print("=" * 50)
    
    claims = scan.get("claims", [])
    claim_verdicts = fact_check.get("claim_verdicts", [])
    verdict_map = {v.get("claim_id"): v for v in claim_verdicts}
    
    for idx, c in enumerate(claims):
        cid = c.get("id")
        assertion = c.get("assertion")
        c_verdict = verdict_map.get(cid, {})
        c_evidence = evidence.get(cid, {})
        
        print(f"\n[Claim #{idx+1}] \"{assertion}\"")
        print(f"  Verdict    : {c_verdict.get('verdict', 'UNVERIFIED').upper()}")
        print(f"  Confidence : {c_verdict.get('confidence_score', 0.5) * 100:.0f}%")
        print(f"  Analysis   : {c_verdict.get('reasoning')}")
        
        # Sources
        sources = c_evidence.get("sources", [])
        if sources:
            print("  Sources Cited:")
            for s in sources:
                s_title = s.get("title", s.get("source", "Reference"))
                s_url = s.get("url", "")
                if s_url:
                    print(f"    - {s_title} ({s_url})")
                else:
                    print(f"    - {s_title}")
    print("=" * 70 + "\n")

def run_cli(headline: str, article: str):
    """
    Executes pipeline and prints formatted reports to terminal.
    """
    logger.info("Initializing Agentic RAG CLI pipeline...")
    
    # 1. API key validation
    if not Settings.validate():
        logger.error("Gemini API key is missing. Process aborted.")
        print("\n[ERROR] Google Gemini API Key is not set.")
        print("Please configure GEMINI_API_KEY in a .env file in the project root, or set it as an environment variable.\n")
        sys.exit(1)
        
    # 2. Build input content
    compound_text = f"Headline: {headline}"
    if article and article.strip():
        compound_text += f"\n\nFull Article Context:\n{article}"
        
    orchestrator = Orchestrator()
    
    print("\nExecuting multi-agent verification pipeline... (Please wait)")
    
    pipeline_result = None
    try:
        # Loop through progress generator and print state updates
        for update in orchestrator.run_pipeline(compound_text):
            status = update["status"]
            message = update["message"]
            
            if status == "error":
                print(f"\n[PIPELINE ERROR] {message}")
                sys.exit(1)
            elif status == "completed":
                pipeline_result = update["data"]
            else:
                print(f" -> {message}")
                
        if pipeline_result:
            display_results(pipeline_result)
        else:
            print("\n[ERROR] Pipeline completed but returned no data.")
            
    except Exception as e:
        logger.exception("An unhandled exception occurred in main runner.")
        print(f"\n[FATAL ERROR] An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    print_banner()
    
    parser = argparse.ArgumentParser(description="VeriTruth Agentic RAG Fake News Detection CLI Runner.")
    parser.add_argument("--headline", "-hl", type=str, help="Headline of the claim or news article.")
    parser.add_argument("--article", "-a", type=str, help="Body text of the news article (optional).")
    parser.add_argument("--interactive", "-i", action="store_true", help="Run the system in interactive prompt mode.")
    
    args = parser.parse_args()
    
    # If no parameters provided or interactive flag chosen, enter interactive mode
    if args.interactive or (not args.headline):
        print("\nEntering Interactive CLI Mode. Press Ctrl+C to exit.")
        try:
            headline = input("\nNews Headline / Claim: ").strip()
            if not headline:
                print("Headline is required. Run aborted.")
                sys.exit(1)
                
            print("Full News Article (optional, press Enter twice to skip):")
            lines = []
            while True:
                line = input()
                if not line:
                    break
                lines.append(line)
            article = "\n".join(lines).strip()
            
            run_cli(headline, article)
        except KeyboardInterrupt:
            print("\nExecution cancelled by user. Goodbye!")
            sys.exit(0)
    else:
        run_cli(args.headline, args.article)

if __name__ == "__main__":
    main()
