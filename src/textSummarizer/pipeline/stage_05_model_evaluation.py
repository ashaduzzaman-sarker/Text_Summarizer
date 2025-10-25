# ============================================================================
# src/textSummarizer/pipeline/stage_05_model_evaluation.py
# ============================================================================
"""Model evaluation pipeline stage."""

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_evaluation import ModelEvaluation
from textSummarizer.logging.logger import logger


class ModelEvaluationPipeline:
    """Pipeline for model evaluation stage."""
    
    def __init__(self):
        """Initialize pipeline."""
        self.stage_name = "Model Evaluation"
    
    def run(self):
        """Execute model evaluation pipeline.
        
        Returns:
            Evaluation metrics
        """
        try:
            logger.info(f">>>>>> Stage: {self.stage_name} started <<<<<<")
            
            # Load configuration
            config_manager = ConfigurationManager()
            model_evaluation_config = config_manager.get_model_evaluation_config()
            
            # Get evaluation parameters
            evaluation_params = config_manager.params.EvaluationArguments
            
            # Run evaluation
            model_evaluation = ModelEvaluation(
                config=model_evaluation_config,
                params=evaluation_params
            )
            metrics = model_evaluation.evaluate()
            
            logger.info(f">>>>>> Stage: {self.stage_name} completed <<<<<<\n")
            return metrics
            
        except Exception as e:
            logger.error(f">>>>>> Stage: {self.stage_name} failed <<<<<<")
            logger.exception(e)
            raise
