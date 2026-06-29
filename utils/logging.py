import logging
import sys
from pathlib import Path
from config.settings import Settings

def setup_logger(name: str = "agentic_rag") -> logging.Logger:
    """Sets up a dual handler logger that logs to stdout and a file."""
    logger = logging.getLogger(name)
    
    # If logger is already configured, return it
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, Settings.LOG_LEVEL, logging.INFO))

    # Formatters
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File Handler
    try:
        log_file = Settings.LOG_FILE
        # Ensure log file parent directories exist
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Failed to initialize file logger: {e}", file=sys.stderr)

    return logger

# Create a default logger instance
logger = setup_logger()
