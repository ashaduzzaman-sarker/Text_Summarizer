import os
import urllib.request as request
import zipfile
from pathlib import Path
from typing import Optional
from textSummarizer.logging.logger import logger
from textSummarizer.utils.common import get_size
from textSummarizer.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_file(self):
        """Download file from URL if it doesn't exist"""
        if not os.path.exists(self.config.local_data_file):
            filename, headers = request.urlretrieve(
                url=self.config.source_URL,
                filename=self.config.local_data_file
            )
            logger.info(f"{filename} downloaded! with following info: \n{headers}")
        else:
            logger.info(f"File already exists of size: {get_size(self.config.local_data_file)}")
    
    def extract_zip_file(self):
        """Extract zip file into the data directory"""
        unzip_path = self.config.unzip_dir
        os.makedirs(unzip_path, exist_ok=True)
        with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
            zip_ref.extractall(unzip_path)
        logger.info(f"Extracted zip file to: {unzip_path}")
    
    def download_huggingface_dataset(self):
        """Download dataset from HuggingFace Hub"""
        try:
            from datasets import load_dataset
            
            logger.info(f"Downloading HuggingFace dataset: {self.config.dataset_name}")
            
            # Load dataset
            dataset = load_dataset(
                self.config.dataset_name,
                split=self.config.dataset_split
            )
            
            # Save dataset to disk
            output_path = self.config.unzip_dir / "huggingface_data"
            os.makedirs(output_path, exist_ok=True)
            
            # Save in different formats
            dataset.save_to_disk(str(output_path))
            logger.info(f"Dataset saved to: {output_path}")
            
            # Optionally save as CSV for easy inspection
            csv_path = output_path / f"{self.config.dataset_name}_{self.config.dataset_split}.csv"
            dataset.to_csv(csv_path)
            logger.info(f"Dataset also saved as CSV: {csv_path}")
            
            # Log dataset info
            logger.info(f"Dataset size: {len(dataset)} examples")
            logger.info(f"Dataset features: {dataset.features}")
            
            return dataset
            
        except ImportError:
            logger.error("datasets library not installed. Install with: pip install datasets")
            raise
        except Exception as e:
            logger.error(f"Error downloading HuggingFace dataset: {str(e)}")
            raise