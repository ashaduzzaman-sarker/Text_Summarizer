from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.logging.logger import logger


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass
    
    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        
        if data_ingestion_config.use_huggingface:
            # Use HuggingFace dataset
            logger.info("Using HuggingFace dataset")
            data_ingestion.download_huggingface_dataset()
        else:
            # Use traditional URL download
            logger.info("Using URL-based download")
            data_ingestion.download_file()
            data_ingestion.extract_zip_file()