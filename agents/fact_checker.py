import json
import re
from typing import List, Dict, Any
from agents.base import BaseAgent
from config.settings import Settings
from config.prompts import VERIFICATION_PROMPT_TEMPLATE
from utils.logging import logger

# Try loading the new google-genai SDK if available
try:
    from google import genai
    from google.genai import types
    HAS_GENAI_SDK = True
    logger.info("[FactCheckerAgent] Successfully loaded new 'google-genai' SDK.")
except ImportError:
    HAS_GENAI_SDK = False
    logger.info("[FactCheckerAgent] 'google-genai' SDK not found. Will fall back to legacy 'google-generativeai'.")

class FactCheckerAgent(BaseAgent):
    def __init__(self):
        """
        Initializes the FactCheckerAgent using custom verification instructions.
        """
        super().__init__(
            name="FactCheckerAgent",
            system_instruction="You are an expert Fact-Checking Agent. Your role is to cross-reference claims against retrieved evidence and return structured classifications."
        )

    def _call_gemini(self, prompt: str) -> str:
        """
        Invokes Gemini model using google-genai SDK if applicable,
        otherwise falls back to the legacy generativeai package wrapper.
        """
        if HAS_GENAI_SDK and Settings.GEMINI_API_KEY:
            try:
                logger.info("[FactCheckerAgent] Calling Gemini using google-genai SDK...")
                client = genai.Client(api_key=Settings.GEMINI_API_KEY)
                response = client.models.generate_content(
                    model=Settings.GEMINI_MODEL,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_instruction,
                        response_mime_type="application/json"
                    )
                )
                if response.text:
                    return response.text
            except Exception as e:
                logger.warning(f"[FactCheckerAgent] google-genai call failed: {e}. Falling back to legacy SDK.")

        # Fallback to base call_llm (google-generativeai)
        return self.call_llm(prompt, json_mode=True)

    def clean_markdown_fences(self, text: str) -> str:
        """
        Removes markdown backticks and code fence tags (such as ```json and ```).
        """
        cleaned = text.strip()
        # Remove leading ```json or ``` with optional whitespaces
        cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned, flags=re.IGNORECASE)
        # Remove trailing ```
        cleaned = re.sub(r'\s*```$', '', cleaned)
        return cleaned.strip()

    def parse_verdict_safely(self, text: str) -> Dict[str, Any]:
        """
        Parses text safely using json.loads, and uses Regex fallback if JSON is malformed.
        Guarantees returned dictionary keys: classification, confidence, reason.
        """
        cleaned = self.clean_markdown_fences(text)
        
        # 1. Try standard JSON parse
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, dict):
                # Normalize keys to lowercase for robust lookup
                normalized = {str(k).lower(): v for k, v in parsed.items()}
                
                classification = normalized.get("classification", normalized.get("verdict", ""))
                confidence = normalized.get("confidence", normalized.get("confidence_score", 0.5))
                reason = normalized.get("reason", normalized.get("reasoning", ""))
                
                try:
                    confidence = float(confidence)
                except (TypeError, ValueError):
                    confidence = 0.5
                    
                return {
                    "classification": str(classification).strip(),
                    "confidence": confidence,
                    "reason": str(reason).strip()
                }
        except Exception as e:
            logger.warning(f"[FactCheckerAgent] json.loads parsing failed: {e}. Attempting Regex fallback.")

        # 2. Fallback Regex Parsing if JSON was malformed
        classification = "Insufficient Evidence"
        confidence = 0.5
        reason = cleaned  # Use raw string as fallback reason
        
        # Extract classification via regex
        class_match = re.search(r'"classification"\s*:\s*"([^"]+)"', cleaned, re.IGNORECASE)
        if class_match:
            classification = class_match.group(1).strip()
        else:
            # Check for direct occurrences of the enum values in the raw string
            if re.search(r'\btrue\b', cleaned, re.IGNORECASE):
                classification = "True"
            elif re.search(r'\bfalse\b', cleaned, re.IGNORECASE):
                classification = "False"
            elif re.search(r'\bmisleading\b', cleaned, re.IGNORECASE):
                classification = "Misleading"
                
        # Extract confidence via regex
        conf_match = re.search(r'"confidence"\s*:\s*([0-9.]+)', cleaned, re.IGNORECASE)
        if conf_match:
            try:
                confidence = float(conf_match.group(1))
            except ValueError:
                confidence = 0.5
                
        # Extract reason via regex
        reason_match = re.search(r'"reason"\s*:\s*"([^"\\]*(?:\\.[^"\\]*)*)"', cleaned, re.IGNORECASE)
        if reason_match:
            reason = reason_match.group(1).strip().replace('\\"', '"')
        else:
            # DOTALL capture fallback
            reason_match_dotall = re.search(r'"reason"\s*:\s*"(.*?)"', cleaned, re.DOTALL | re.IGNORECASE)
            if reason_match_dotall:
                reason = reason_match_dotall.group(1).strip()
                
        return {
            "classification": classification,
            "confidence": confidence,
            "reason": reason
        }

    def verify_claim(self, claim: str, evidence: str) -> Dict[str, Any]:
        """
        Verifies a single claim against retrieved evidence context.
        Ensures a secure, non-crashing dictionary return format.
        """
        logger.info(f"[FactCheckerAgent] Verifying claim: '{claim[:50]}...'")
        
        prompt = VERIFICATION_PROMPT_TEMPLATE.format(claim=claim, evidence=evidence)
        
        # Initialize default return values in case of complete API failure
        default_return = {
            "classification": "Insufficient Evidence",
            "confidence": 0.0,
            "reason": "An error occurred during Gemini API verification calls."
        }
        
        try:
            raw_response = self._call_gemini(prompt)
            if not raw_response:
                logger.error("[FactCheckerAgent] Received empty response from model.")
                return default_return
                
            # Parse result safely
            result = self.parse_verdict_safely(raw_response)
            
            # Enforce validation on classification enums
            valid_classes = {"True", "False", "Misleading", "Insufficient Evidence"}
            # Check case-insensitive match and correct the casing
            found_class = "Insufficient Evidence"
            for vc in valid_classes:
                if result["classification"].lower() == vc.lower():
                    found_class = vc
                    break
            result["classification"] = found_class
            
            # Enforce confidence limits
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            
            return result
            
        except Exception as e:
            logger.error(f"[FactCheckerAgent] Fatal exception during verify_claim: {e}")
            return {
                "classification": "Insufficient Evidence",
                "confidence": 0.0,
                "reason": f"System verification call error: {str(e)}"
            }

    def verify_claims(self, claims: List[Dict], evidence_by_claim: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Orchestrates verification for multiple claims and maps output structures.
        Preserves fully backward-compatible signatures.
        """
        logger.info("[FactCheckerAgent] Running multi-claim verification processing...")
        claim_verdicts = []
        classifications = []
        
        for index, claim in enumerate(claims):
            claim_id = claim.get("id", f"claim_{index}")
            assertion = claim.get("assertion", "")
            evidence_info = evidence_by_claim.get(claim_id, {})
            evidence_text = evidence_info.get("synthesized_evidence", "No evidence available.")
            
            result = self.verify_claim(assertion, evidence_text)
            
            claim_verdicts.append({
                "claim_id": claim_id,
                "assertion": assertion,
                "verdict": result["classification"],
                "reasoning": result["reason"],
                "confidence_score": result["confidence"],
                "citations": [src.get("url", "") for src in evidence_info.get("sources", []) if src.get("url")]
            })
            classifications.append(result["classification"])

        # Aggregate overall status
        if "False" in classifications:
            overall_verdict = "False"
        elif "Misleading" in classifications:
            overall_verdict = "Misleading"
        elif classifications and all(c == "True" for c in classifications):
            overall_verdict = "True"
        else:
            overall_verdict = "Insufficient Evidence"

        reasons_list = [f"Claim {i+1} classification is {v['verdict']}." for i, v in enumerate(claim_verdicts)]
        overall_reasoning = f"Analysis finalized. Verdict: {overall_verdict}. Details: " + " ".join(reasons_list)

        return {
            "overall_verdict": overall_verdict,
            "overall_reasoning": overall_reasoning,
            "claim_verdicts": claim_verdicts
        }
