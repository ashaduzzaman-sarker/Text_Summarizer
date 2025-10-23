# ============================================================================
# src/textSummarizer/logging/logger.py
# ============================================================================
import os
import sys
import logging
from pathlib import Path

# Logging configuration
logging_str = "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
log_dir = "logs"
log_filepath = os.path.join(log_dir, "running_logs.log")

# Create logs directory
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=logging_str,
    handlers=[
        logging.FileHandler(log_filepath),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("textSummarizerLogger")


# Optional: Utility function to log exceptions
def log_exception(exc: Exception, context: str = ""):
    """Log exception with context"""
    if context:
        logger.error(f"{context}: {str(exc)}", exc_info=True)
    else:
        logger.error(str(exc), exc_info=True)


# Optional: Context manager for logging function execution
class LogExecutionTime:
    """Context manager to log execution time of code blocks"""
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
    
    def __enter__(self):
        import time
        self.start_time = time.time()
        logger.info(f"Starting: {self.operation_name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        import time
        elapsed = time.time() - self.start_time
        if exc_type is None:
            logger.info(f"Completed: {self.operation_name} (took {elapsed:.2f}s)")
        else:
            logger.error(f"Failed: {self.operation_name} (took {elapsed:.2f}s)")
        return False