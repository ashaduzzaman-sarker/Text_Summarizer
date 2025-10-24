# ============================================================================
# main.py
# ============================================================================
"""Main execution script."""

from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationPipeline
from textSummarizer.logging.logger import logger


if __name__ == "__main__":
    # # Stage 1: Data Ingestion
    # try:
    #     logger.info("=" * 60)
    #     pipeline = DataIngestionPipeline()
    #     pipeline.run()
    # except Exception as e:
    #     logger.error("Data Ingestion failed")
    #     raise
    
    # # Stage 2: Data Validation
    # try:
    #     logger.info("=" * 60)
    #     pipeline = DataValidationPipeline()
    #     validation_passed = pipeline.run()
        
    #     if not validation_passed:
    #         logger.error("Data validation failed - check status file")
    #         raise ValueError("Data validation checks failed")
        
    # except Exception as e:
    #     logger.error("Data Validation failed")
    #     raise
    
    # Stage 3: Data Transformation
    try:
        logger.info("=" * 60)
        pipeline = DataTransformationPipeline()
        pipeline.run()
    except Exception as e:
        logger.error("Data Transformation failed")
        raise
    
    logger.info("=" * 60)
    logger.info("All pipeline stages completed successfully!")