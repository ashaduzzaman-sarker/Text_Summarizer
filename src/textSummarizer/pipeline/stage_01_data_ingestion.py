# ============================================================================
# src/textSummarizer/pipeline/stage_01_data_ingestion.py
# ============================================================================
"""Data ingestion pipeline stage."""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.logging.logger import logger

class DataIngestionPipeline:
    """Pipeline for data ingestion stage."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.stage_name = "Data Ingestion"
    
    def run(self):
        """Execute data ingestion pipeline.
        
        Returns:
            Loaded dataset
        """
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")
            
            # Load configuration
            config_manager = ConfigurationManager()
            data_ingestion_config = config_manager.get_data_ingestion_config()
            
            # Run data ingestion
            data_ingestion = DataIngestion(config=data_ingestion_config)
            dataset = data_ingestion.download_dataset()
            
            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return dataset
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise