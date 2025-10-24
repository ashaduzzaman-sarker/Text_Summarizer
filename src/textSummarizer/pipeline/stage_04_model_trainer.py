# ============================================================================
# src/textSummarizer/pipeline/stage_04_model_trainer.py
# ============================================================================
"""Model training pipeline stage."""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer
from textSummarizer.logging.logger import logger


class ModelTrainerPipeline:
    """Pipeline for model training stage."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.stage_name = "Model Training"
    
    def run(self):
        """Execute model training pipeline.
        
        Returns:
            Training result
        """
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")
            
            # Load configuration
            config_manager = ConfigurationManager()
            model_trainer_config = config_manager.get_model_trainer_config()
            
            # Get training parameters
            training_params = config_manager.params.TrainingArguments
            
            # Run training
            model_trainer = ModelTrainer(
                config=model_trainer_config,
                params=training_params
            )
            train_result = model_trainer.train()
            
            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return train_result
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise