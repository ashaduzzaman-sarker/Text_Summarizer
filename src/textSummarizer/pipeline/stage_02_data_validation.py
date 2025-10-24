# ============================================================================
# src/textSummarizer/pipeline/stage_02_data_validation.py
# ============================================================================
"""Data validation pipeline stage."""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_validation import DataValidation
from textSummarizer.logging.logger import logger


class DataValidationPipeline:
    """Pipeline for data validation stage."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.stage_name = "Data Validation"
    
    def run(self) -> bool:
        """Execute data validation pipeline.
        
        Returns:
            True if validation passes, False otherwise
        """
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")
            
            # Load configuration
            config_manager = ConfigurationManager()
            data_validation_config = config_manager.get_data_validation_config()
            
            # Run validation
            data_validation = DataValidation(config=data_validation_config)
            validation_passed = data_validation.validate()
            
            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return validation_passed
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise