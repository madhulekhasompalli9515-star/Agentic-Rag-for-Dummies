import time
from typing import List, Dict
from duckduckgo_search import DDGS
from utils.logging import logger

def web_search(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    Perform a search on DuckDuckGo using duckduckgo-search.
    Returns a list of dictionaries with title, url, and snippet.
    """
    logger.info(f"Searching for query: '{query}'")
    results = []
    
    # Try with retries and exponential backoff
    for attempt in range(3):
        try:
            with DDGS() as ddgs:
                # ddgs.text performs text search and returns list of dicts
                search_results = list(ddgs.text(query, max_results=max_results))
                
                for r in search_results:
                    results.append({
                        "title": r.get("title", "No Title"),
                        "url": r.get("href", ""),
                        "snippet": r.get("body", "")
                    })
                break  # Success
        except Exception as e:
            logger.warning(f"DuckDuckGo search attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2 ** attempt)  # Backoff
            else:
                logger.error(f"DuckDuckGo search failed permanently for query: '{query}'")
                
    return results
