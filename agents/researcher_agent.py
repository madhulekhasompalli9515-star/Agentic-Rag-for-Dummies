import requests
from bs4 import BeautifulSoup
import urllib.parse
import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from duckduckgo_search import DDGS
from agents.base import BaseAgent
from config.prompts import RESEARCHER_SYSTEM_INSTRUCTION
from utils.logging import logger
from config.settings import Settings

# =====================================================================
# SEARCH PROVIDER INTERFACE
# =====================================================================
class SearchProvider(ABC):
    """
    Abstract interface for search providers.
    Allows swapping the scraping-based search with an API-based search (e.g. Tavily, Serper) later.
    """
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Executes search query and returns a list of dictionaries with:
        source: Name of the website/source
        url: Link to the source article
        snippet: Snip/summary text from search output
        """
        pass

# =====================================================================
# REQUESTS + BEAUTIFUL SOUP SCRAPER PROVIDER
# =====================================================================
class DuckDuckGoScraperProvider(SearchProvider):
    """
    Legacy scraper implementation of SearchProvider using requests and BeautifulSoup.
    Kept as a secondary fallback.
    """
    TRUSTED_DOMAINS = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
        "snopes.com", "factcheck.org", "politifact.com", "npr.org", 
        "theguardian.com", "washingtonpost.com"
    ]

    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9"
        }

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        logger.info(f"[DuckDuckGoScraperProvider] Running requests search for: '{query}'")
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        results = []
        seen_urls = set()
        try:
            response = requests.get(url, params=params, headers=self.headers, timeout=10)
            if response.status_code != 200:
                logger.warning(f"[DuckDuckGoScraperProvider] Received non-200 status code: {response.status_code}")
                return []
            soup = BeautifulSoup(response.text, "html.parser")
            result_divs = soup.find_all("div", class_="result")
            for div in result_divs:
                a_tag = div.find("a", class_="result__a")
                snippet_tag = div.find("a", class_="result__snippet")
                if a_tag:
                    title = a_tag.text.strip()
                    raw_href = a_tag.get("href", "")
                    extracted_url = raw_href
                    if "/l/?uddg=" in raw_href:
                        parsed_href = urllib.parse.urlparse(raw_href)
                        query_params = urllib.parse.parse_qs(parsed_href.query)
                        if "uddg" in query_params:
                            extracted_url = query_params["uddg"][0]
                    clean_url = urllib.parse.unquote(extracted_url)
                    if clean_url in seen_urls:
                        continue
                    seen_urls.add(clean_url)
                    snippet = snippet_tag.text.strip() if snippet_tag else "No summary available."
                    domain = urllib.parse.urlparse(clean_url).netloc
                    source_name = domain.replace("www.", "")
                    results.append({
                        "title": title,
                        "url": clean_url,
                        "snippet": snippet,
                        "source": source_name
                    })
            trusted_results = [r for r in results if any(td in r["url"].lower() for td in self.TRUSTED_DOMAINS)]
            general_results = [r for r in results if not any(td in r["url"].lower() for td in self.TRUSTED_DOMAINS)]
            sorted_results = trusted_results + general_results
            return sorted_results[:max_results]
        except Exception as e:
            logger.warning(f"[DuckDuckGoScraperProvider] Scraper search failed: {e}")
            return []

class DuckDuckGoSearchProvider(SearchProvider):
    """
    Search provider using the duckduckgo-search Python package.
    Highly reliable, uses API-like endpoints, doesn't get blocked by HTML CAPTCHAs.
    """
    TRUSTED_DOMAINS = [
        "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com",
        "snopes.com", "factcheck.org", "politifact.com", "npr.org", 
        "theguardian.com", "washingtonpost.com"
    ]

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        logger.info(f"[DuckDuckGoSearchProvider] Querying duckduckgo-search package for: '{query}'")
        results = []
        try:
            with DDGS() as ddgs:
                # Query using package text search
                search_results = list(ddgs.text(query, max_results=max_results * 3))
                for r in search_results:
                    title = r.get("title", "No Title")
                    url = r.get("href", "")
                    snippet = r.get("body", "No snippet available.")
                    if not url:
                        continue
                    domain = urllib.parse.urlparse(url).netloc
                    source_name = domain.replace("www.", "")
                    results.append({
                        "title": title,
                        "url": url,
                        "URL": url,
                        "snippet": snippet,
                        "source": source_name
                    })
            # Sort to prioritize trusted domains
            trusted = [r for r in results if any(td in r["url"].lower() for td in self.TRUSTED_DOMAINS)]
            general = [r for r in results if not any(td in r["url"].lower() for td in self.TRUSTED_DOMAINS)]
            sorted_results = trusted + general
            logger.info(f"[DuckDuckGoSearchProvider] Retrieved {len(sorted_results[:max_results])} results.")
            return sorted_results[:max_results]
        except Exception as e:
            logger.error(f"[DuckDuckGoSearchProvider] Search package query failed: {e}")
            return []

class TavilySearchProvider(SearchProvider):
    """
    Search provider using Tavily API client via HTTP requests.
    Used when Settings.TAVILY_API_KEY is configured.
    """
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        logger.info(f"[TavilySearchProvider] Querying Tavily API for: '{query}'")
        try:
            response = requests.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "search_depth": "basic"
                },
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                results = []
                for r in data.get("results", []):
                    url = r.get("url", "")
                    domain = urllib.parse.urlparse(url).netloc
                    source_name = domain.replace("www.", "")
                    results.append({
                        "title": r.get("title", "No Title"),
                        "url": url,
                        "URL": url,
                        "snippet": r.get("content", ""),
                        "source": source_name
                    })
                logger.info(f"[TavilySearchProvider] Retrieved {len(results)} results.")
                return results
            else:
                logger.error(f"[TavilySearchProvider] Tavily returned status {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"[TavilySearchProvider] Search query failed: {e}")
            return []

# =====================================================================
# RESEARCHER AGENT
# =====================================================================
class ResearcherAgent(BaseAgent):
    def __init__(self, provider: SearchProvider = None):
        """
        Initializes the Researcher Agent.
        Uses a pluggable SearchProvider.
        Defaults to Tavily if TAVILY_API_KEY is configured, otherwise DuckDuckGoSearchProvider.
        """
        super().__init__(
            name="ResearcherAgent",
            system_instruction=RESEARCHER_SYSTEM_INSTRUCTION
        )
        if provider:
            self.provider = provider
        elif Settings.TAVILY_API_KEY:
            self.provider = TavilySearchProvider(Settings.TAVILY_API_KEY)
        else:
            self.provider = DuckDuckGoSearchProvider()
        
        # Keep scraper provider as a fallback reference
        self.fallback_provider = DuckDuckGoScraperProvider()

    def search_evidence(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """
        Exposes web searching capability using the pluggable provider,
        falling back to alternative searchers if no evidence is retrieved.
        """
        results = self.provider.search(query, max_results=max_results)
        if not results and self.provider != self.fallback_provider:
            logger.warning("[ResearcherAgent] Primary search returned no results. Attempting fallback scraper search...")
            results = self.fallback_provider.search(query, max_results=max_results)
        return results

    def research_claim(self, claim_assertion: str, claim_context: str, search_results: List[Dict[str, str]]) -> dict:
        """
        Takes search results/evidence and synthesizes them into an evidence context report using Gemini.
        """
        logger.info(f"[ResearcherAgent] Synthesizing research findings for claim: '{claim_assertion}'...")
        
        # Format the evidence documents
        formatted_results = ""
        for i, res in enumerate(search_results):
            formatted_results += f"\n--- Source #{i+1} ---\nSource: {res.get('source', 'Unknown')}\nURL: {res.get('url')}\nContent: {res.get('snippet')}\n"
            
        prompt = f"""
Claim to verify: "{claim_assertion}"
Context: "{claim_context}"

Here is the search evidence retrieved:
{formatted_results if formatted_results else "No search evidence retrieved."}

Synthesize these findings and return structured JSON matching the instructions schema:
"""
        try:
            raw_response = self.call_llm(prompt, json_mode=True)
            cleaned = raw_response.strip()
            
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()
            
            parsed_data = json.loads(cleaned)
            
            # Simple Schema Validation
            if "synthesized_evidence" not in parsed_data:
                parsed_data["synthesized_evidence"] = "No evidence synthesized."
            if "key_findings" not in parsed_data:
                parsed_data["key_findings"] = []
            if "sources" not in parsed_data:
                parsed_data["sources"] = []
                
            # Map source list formatting from scraping structure to prompts structure
            # (i.e. ensure it contains title, url, snippet)
            for idx, src in enumerate(parsed_data["sources"]):
                if "title" not in src:
                    src["title"] = src.get("source", "Reference Source")
                    
            logger.info(f"[ResearcherAgent] Synthesized report completed successfully.")
            return parsed_data
            
        except json.JSONDecodeError as jde:
            logger.error(f"[ResearcherAgent] Failed to decode LLM response JSON: {jde}")
            return {
                "synthesized_evidence": "Failed to synthesize evidence due to formatting error.",
                "key_findings": ["JSON decode error"],
                "sources": [{"title": s.get("source", "Source"), "url": s.get("url"), "snippet": s.get("snippet")} for s in search_results]
            }
        except Exception as e:
            logger.error(f"[ResearcherAgent] Error in research_claim: {e}")
            raise e
