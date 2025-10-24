# ============================================================================
# src/textSummarizer/entity/config_entity.py
# ============================================================================
"""Configuration management."""

from pathlib import Path
from textSummarizer.constants.constants import CONFIG_FILE_PATH
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    """Manages configuration loading and directory creation."""
    
    def __init__(self, config_path: Path = CONFIG_FILE_PATH):
        """Initialize configuration manager.
        
        Args:
            config_path: Path to config.yaml file
        """
        self.config = read_yaml(config_path)
        create_directories([Path(self.config.artifacts_root)])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration.
        
        Returns:
            DataIngestionConfig instance
        """
        config = self.config.data_ingestion
        
        # Create root directory
        create_directories([Path(config.root_dir)])
        
        return DataIngestionConfig(
            root_dir=Path(config.root_dir),
            cache_dir=Path(config.cache_dir),
            dataset_name=config.dataset_name,
            config_name=config.config_name,
            split=config.split,
            max_samples=config.get('max_samples')
        )