from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.logging.logger import logger

STAGE_NAME = "Data Ingestion Stage"


def main():
    """Execute the data ingestion pipeline."""
    try:
        logger.info(f">>>>>> {STAGE_NAME} started <<<<<<")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.error(f">>>>>> {STAGE_NAME} failed <<<<<<")
        logger.exception(e)
        raise


if __name__ == "__main__":
    main()
    
# ============================================================================
# main.py
# ============================================================================
# from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
# from textSummarizer.logging.logger import logger

# STAGE_NAME = "Data Ingestion stage"

# try:
#     logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
#     data_ingestion = DataIngestionTrainingPipeline()
#     data_ingestion.main()
#     logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
# except Exception as e:
#     logger.exception(e)
#     raise e