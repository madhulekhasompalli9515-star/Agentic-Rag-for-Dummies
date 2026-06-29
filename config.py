# =====================================================================
# NOTE ON PYTHON NAMESPACE SHADOWING:
# To prevent Python from resolving the directory 'config/' instead of 
# this file during module imports, the core loader is implemented in 
# 'app_config.py'.
#
# Always use: import app_config
# =====================================================================

from app_config import (
    GEMINI_API_KEY,
    TAVILY_API_KEY,
    GEMINI_MODEL,
    EMBEDDING_MODEL,
    LOG_LEVEL,
    LOG_FILE,
    VECTOR_DB_DIR,
    MAX_SEARCH_RESULTS,
    validate_config,
    set_api_key
)
