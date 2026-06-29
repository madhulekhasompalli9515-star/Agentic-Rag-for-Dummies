import sys
from pathlib import Path

# Add project root to sys.path to ensure app_config.py is resolvable
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import app_config

class Settings:
    # Google Gemini Configuration
    GEMINI_API_KEY = app_config.GEMINI_API_KEY
    TAVILY_API_KEY = app_config.TAVILY_API_KEY
    GEMINI_MODEL = app_config.GEMINI_MODEL
    EMBEDDING_MODEL = app_config.EMBEDDING_MODEL

    # Logging
    LOG_LEVEL = app_config.LOG_LEVEL
    LOG_FILE = app_config.LOG_FILE

    # Vector DB
    VECTOR_DB_DIR = app_config.VECTOR_DB_DIR

    # Search Configuration
    MAX_SEARCH_RESULTS = app_config.MAX_SEARCH_RESULTS

    @classmethod
    def validate(cls) -> bool:
        """Check if API key is present and not a placeholder."""
        cls.GEMINI_API_KEY = app_config.GEMINI_API_KEY
        return app_config.validate_config()

    @classmethod
    def set_api_key(cls, api_key: str):
        """Set the API key dynamically."""
        app_config.set_api_key(api_key)
        cls.GEMINI_API_KEY = api_key
