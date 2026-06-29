import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env file
load_dotenv(dotenv_path=BASE_DIR / ".env")

# Expose Secure Variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = BASE_DIR / "app.log"

# RAG Database Configuration
VECTOR_DB_DIR = os.getenv("VECTOR_DB_DIR", str(BASE_DIR / "data" / "vector_db"))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", "5"))

def validate_config() -> bool:
    """
    Checks that the Google Gemini API key has been specified
    and is not the default placeholder.
    """
    if not GEMINI_API_KEY.strip():
        return False
    if GEMINI_API_KEY == "your_gemini_api_key_here":
        return False
    return True

def set_api_key(api_key: str):
    """Dynamically sets the API key in environment variables."""
    global GEMINI_API_KEY
    GEMINI_API_KEY = api_key
    os.environ["GEMINI_API_KEY"] = api_key
