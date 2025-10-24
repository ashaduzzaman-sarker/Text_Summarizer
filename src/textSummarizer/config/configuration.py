# ============================================================================
# src/textSummarizer/entity/config_entity.py
# ============================================================================
"""Configuration management."""

from pathlib import Path
from textSummarizer.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)


class ConfigurationManager:
    """Manages configuration loading and directory creation."""
    
    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH
    ):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file
            params_path: Path to params.yaml file
        """
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([Path(self.config.artifacts_root)])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration."""
        config = self.config.data_ingestion
        create_directories([Path(config.root_dir)])
        
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            cache_dir=Path(config.cache_dir),
            dataset_name=config.dataset_name,
            config_name=config.config_name,
            split=config.split,
            max_samples=config.get('max_samples')
        )
    
    def get_data_validation_config(self) -> DataValidationConfig:
        """Get data validation configuration."""
        config = self.config.data_validation
        create_directories([Path(config.root_dir)])
        
        return DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            data_dir=Path(config.data_dir),
            required_columns=config.required_columns,
            min_samples=config.min_samples
        )
    
    def get_data_transformation_config(self) -> DataTransformationConfig:
        """Get data transformation configuration."""
        config = self.config.data_transformation
        create_directories([Path(config.root_dir)])
        
        return DataTransformationConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir),
            tokenizer_name=config.tokenizer_name,
            max_input_length=config.max_input_length,
            max_target_length=config.max_target_length,
            padding=config.padding,
            batch_size=config.batch_size
        )
    
    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """Get model trainer configuration.
        
        Returns:
            ModelTrainerConfig instance
        """
        config = self.config.model_trainer
        create_directories([Path(config.root_dir)])
        
        return ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            data_dir=Path(config.data_dir),
            tokenizer_dir=Path(config.tokenizer_dir),
            model_name=config.model_name,
            train_split=config.train_split,
            seed=config.seed
        )
