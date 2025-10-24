from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.logging.logger import logger

class DataIngestionTrainingPipeline:
    """Pipeline for data ingestion stage."""
    
    def __init__(self):
        """Initialize pipeline without requiring config parameter."""
        self.stage_name = "Data Ingestion"
    
    def main(self):
        """Execute data ingestion pipeline."""
        config_manager = ConfigurationManager()
        data_ingestion_config = config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        result = data_ingestion.ingest()
        return result

# ============================================================================
# src/textSummarizer/pipeline/stage_01_data_ingestion.py
# ============================================================================
# from textSummarizer.config.configuration import ConfigurationManager
# from textSummarizer.components.data_ingestion import DataIngestion
# from textSummarizer.logging.logger import logger


# class DataIngestionTrainingPipeline:
#     def __init__(self):
#         pass
    
#     def main(self):
#         config = ConfigurationManager()
#         data_ingestion_config = config.get_data_ingestion_config()
#         data_ingestion = DataIngestion(config=data_ingestion_config)
        
#         if data_ingestion_config.use_huggingface:
#             # Use HuggingFace dataset
#             logger.info("Using HuggingFace dataset")
#             data_ingestion.download_huggingface_dataset()
#         else:
#             # Use traditional URL download
#             logger.info("Using URL-based download")
#             data_ingestion.download_file()
#             data_ingestion.extract_zip_file()