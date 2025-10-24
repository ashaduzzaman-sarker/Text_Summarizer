# ============================================================================
# src/textSummarizer/pipeline/stage_03_data_transformation.py
# ============================================================================
"""Data transformation pipeline stage."""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_transformation import DataTransformation
from textSummarizer.logging.logger import logger


class DataTransformationPipeline:
    """Pipeline for data transformation stage."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.stage_name = "Data Transformation"
    
    def run(self):
        """Execute data transformation pipeline.
        
        Returns:
            Transformed dataset
        """
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")
            
            # Load configuration
            config_manager = ConfigurationManager()
            data_transformation_config = config_manager.get_data_transformation_config()
            
            # Run transformation
            data_transformation = DataTransformation(config=data_transformation_config)
            transformed_dataset = data_transformation.transform()
            
            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return transformed_dataset
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise
