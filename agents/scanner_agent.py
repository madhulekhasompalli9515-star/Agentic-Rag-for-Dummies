import re
import json
from typing import Dict, Any, List
from agents.base import BaseAgent
from config.prompts import SCANNER_SYSTEM_INSTRUCTION
from utils.logging import logger

class ScannerAgent(BaseAgent):
    def __init__(self):
        """
        Initializes the ScannerAgent using the Scanner system instructions.
        """
        super().__init__(
            name="ScannerAgent",
            system_instruction=SCANNER_SYSTEM_INSTRUCTION
        )

    def clean_text(self, text: str) -> str:
        """
        Cleans the input text by:
        1. Validating that it is not empty.
        2. Converting to lowercase.
        3. Removing special characters (keeping alphanumeric, basic punctuation, and hyphens).
        4. Removing extra whitespace.
        """
        if not text or not text.strip():
            logger.error("[ScannerAgent] Input validation failed: Empty text provided.")
            raise ValueError("Input news text cannot be empty.")
            
        # Convert to lowercase
        cleaned = text.lower()
        
        # Remove special characters (keep only letters, numbers, spaces, periods, commas, exclamation marks, question marks, and hyphens)
        cleaned = re.sub(r'[^a-z0-9\s\.\,\!\?\-]', '', cleaned)
        
        # Remove extra spaces and strip leading/trailing spaces
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        logger.info("[ScannerAgent] Successfully completed preprocessing and text cleaning.")
        return cleaned

    def extract_keywords_and_entities(self, cleaned_text: str) -> Dict[str, Any]:
        """
        Uses the Gemini model to extract keywords and named entities (Persons, Organizations, Locations)
        from the cleaned text.
        """
        logger.info("[ScannerAgent] Extracting keywords and named entities...")
        
        prompt = f"""
Analyze the following text and extract:
1. A list of important keywords.
2. A list of Named Entities categorized into "persons", "organizations", and "locations".

You must respond ONLY in structured JSON format matching this schema:
{{
  "keywords": ["word1", "word2"],
  "entities": {{
    "persons": ["Name 1"],
    "organizations": ["Org 1"],
    "locations": ["Location 1"]
  }}
}}
Do not wrap your output in ```json markdown blocks, output raw JSON.

Text to analyze:
"{cleaned_text}"
"""
        try:
            raw_response = self.call_llm(prompt, json_mode=True)
            cleaned = raw_response.strip()
            
            # Strip markdown json code block signs if returned
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            extracted = json.loads(cleaned)
            
            # Ensure keys exist
            if "keywords" not in extracted:
                extracted["keywords"] = []
            if "entities" not in extracted:
                extracted["entities"] = {"persons": [], "organizations": [], "locations": []}
                
            return extracted
            
        except json.JSONDecodeError as jde:
            logger.error(f"[ScannerAgent] JSON decoding failed during keyword/entity extraction: {jde}")
            return {
                "keywords": [],
                "entities": {"persons": [], "organizations": [], "locations": []}
            }
        except Exception as e:
            logger.error(f"[ScannerAgent] Error in extract_keywords_and_entities: {e}")
            raise e

    def scan_article(self, text: str) -> Dict[str, Any]:
        """
        Integrates cleaning, extraction, and claims analysis into a single entry point.
        This maintains full backward-compatibility with the orchestrator pipeline.
        """
        logger.info("[ScannerAgent] Starting article scan workflow...")
        
        # 1. Clean input
        cleaned_text = self.clean_text(text)
        
        # 2. Extract keywords & entities
        extraction_data = self.extract_keywords_and_entities(cleaned_text)
        
        # 3. Perform the standard claim and sensationalism scan
        prompt = f"Analyze the following text and return the claims, sensationalism score, and list of fallacies in raw JSON format:\n\n{cleaned_text}"
        
        try:
            raw_response = self.call_llm(prompt, json_mode=True)
            cleaned = raw_response.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            parsed_data = json.loads(cleaned)
            
            # Inject keywords and entities into the final output
            parsed_data["cleaned_text"] = cleaned_text
            parsed_data["keywords"] = extraction_data.get("keywords", [])
            parsed_data["entities"] = extraction_data.get("entities", {"persons": [], "organizations": [], "locations": []})
            
            # Simple validation of keys
            if "claims" not in parsed_data:
                parsed_data["claims"] = []
            if "sensationalism_score" not in parsed_data:
                parsed_data["sensationalism_score"] = 0
            if "biases_and_fallacies" not in parsed_data:
                parsed_data["biases_and_fallacies"] = []
                
            logger.info(f"[ScannerAgent] Scan complete. Found {len(parsed_data['claims'])} claims, {len(parsed_data['keywords'])} keywords.")
            return parsed_data
            
        except json.JSONDecodeError as jde:
            logger.error(f"[ScannerAgent] Failed to decode JSON from response: {jde}.")
            return {
                "cleaned_text": cleaned_text,
                "keywords": extraction_data.get("keywords", []),
                "entities": extraction_data.get("entities", {}),
                "sensationalism_score": 5,
                "biases_and_fallacies": ["Failed to parse claims JSON"],
                "claims": []
            }
        except Exception as e:
            logger.error(f"[ScannerAgent] Error in scan_article: {e}")
            raise e
