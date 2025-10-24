# ============================================================================
# src/textSummarizer/components/data_ingestion.py
# ============================================================================
"""Data ingestion component for loading datasets."""

from pathlib import Path
from textSummarizer.logging.logger import logger
from textSummarizer.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """Handles data ingestion from HuggingFace datasets."""
    
    def __init__(self, config: DataIngestionConfig):
        """Initialize with configuration.
        
        Args:
            config: DataIngestionConfig instance
        """
        self.config = config
        self._validate_config()
    
    def _validate_config(self) -> None:
        """Validate that required configuration is present."""
        if not self.config.dataset_name:
            raise ValueError("dataset_name is required in configuration")
        
        # Create directories
        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self):
        """Download and save dataset from HuggingFace.
        
        Returns:
            Dataset object
            
        Raises:
            ImportError: If datasets library is not installed
            Exception: For any download/processing errors
        """
        try:
            from datasets import load_dataset
        except ImportError as e:
            logger.error("datasets library not installed")
            raise ImportError(
                "Please install: pip install datasets"
            ) from e
        
        try:
            logger.info(f"Loading dataset: {self.config.dataset_name}")
            logger.info(f"Config: {self.config.config_name}, Split: {self.config.split}")
            
            # Load dataset
            dataset = load_dataset(
                self.config.dataset_name,
                name=self.config.config_name,
                split=self.config.split,
                cache_dir=str(self.config.cache_dir)
            )
            
            # Limit samples if specified (useful for testing)
            if self.config.max_samples:
                original_size = len(dataset)
                dataset = dataset.select(
                    range(min(self.config.max_samples, len(dataset)))
                )
                logger.info(f"Limited dataset: {original_size} -> {len(dataset)} samples")
            
            # Log dataset info
            logger.info(f"Dataset loaded: {len(dataset)} samples")
            logger.info(f"Features: {list(dataset.features.keys())}")
            
            # Save dataset to disk
            output_path = self.config.root_dir / "dataset"
            dataset.save_to_disk(str(output_path))
            logger.info(f"Dataset saved to: {output_path}")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to download dataset: {e}")
            raise