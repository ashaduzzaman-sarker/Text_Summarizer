# src/textSummarizer/logging/logger.py

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

## Example Usage
# from textSummarizer.logging import logger

## Basic logging (your current usage)
# logger.info("Starting data ingestion")
# logger.warning("Model file not found, downloading...")