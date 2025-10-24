# ============================================================================
# main.py
# ============================================================================
"""Main execution script."""

from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from textSummarizer.logging.logger import logger


if __name__ == "__main__":
    try:
        pipeline = DataIngestionPipeline()
        pipeline.run()
    except Exception as e:
        logger.error("Pipeline execution failed")
        raise