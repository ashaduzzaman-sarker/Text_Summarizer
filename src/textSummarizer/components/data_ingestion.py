import os
import zipfile
import requests
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from textSummarizer.logging.logger import logger
from textSummarizer.utils.common import get_size
from textSummarizer.entity.config_entity import DataIngestionConfig


class DataIngestion:
    """Unified data ingestion from HuggingFace, Kaggle, and URLs."""
    
    def __init__(self, config: DataIngestionConfig):
        """Initialize with configuration.
        
        Args:
            config: DataIngestionConfig instance
        """
        self.config = config
        self.config.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def ingest(self):
        """Main ingestion method that routes to appropriate source.
        
        Returns:
            Dataset or path to downloaded data
        """
        logger.info(f"Starting data ingestion from {self.config.source_type}")
        
        if self.config.source_type == "huggingface":
            return self.download_huggingface_dataset()
        elif self.config.source_type == "kaggle":
            return self.download_kaggle_dataset()
        elif self.config.source_type == "url":
            return self.download_from_url()
        else:
            raise ValueError(f"Unknown source type: {self.config.source_type}")
    
    def download_huggingface_dataset(self):
        """Download and process HuggingFace dataset.
        
        Returns:
            Loaded dataset
        """
        try:
            from datasets import load_dataset
            
            if not self.config.hf_dataset_name:
                raise ValueError("hf_dataset_name is required for HuggingFace source")
            
            logger.info(f"Loading HuggingFace dataset: {self.config.hf_dataset_name}")
            
            # Load dataset
            dataset = load_dataset(
                self.config.hf_dataset_name,
                name=self.config.hf_config_name,
                split=self.config.hf_split,
                streaming=self.config.hf_streaming,
                cache_dir=str(self.config.cache_dir)
            )
            
            # Limit samples if specified
            if self.config.max_samples and not self.config.hf_streaming:
                dataset = dataset.select(range(min(self.config.max_samples, len(dataset))))
                logger.info(f"Limited to {len(dataset)} samples")
            
            # Save dataset
            output_path = self.config.root_dir / "dataset"
            
            if self.config.output_format == "arrow":
                dataset.save_to_disk(str(output_path))
                logger.info(f"Saved Arrow dataset to {output_path}")
            elif self.config.output_format == "parquet":
                output_file = output_path.with_suffix('.parquet')
                dataset.to_parquet(str(output_file))
                logger.info(f"Saved Parquet to {output_file}")
            elif self.config.output_format == "csv":
                output_file = output_path.with_suffix('.csv')
                dataset.to_csv(str(output_file))
                logger.info(f"Saved CSV to {output_file}")
            elif self.config.output_format == "json":
                output_file = output_path.with_suffix('.json')
                dataset.to_json(str(output_file))
                logger.info(f"Saved JSON to {output_file}")
            
            if not self.config.hf_streaming:
                logger.info(f"Dataset info: {len(dataset)} samples")
                logger.info(f"Features: {list(dataset.features.keys())}")
            
            return dataset
            
        except ImportError:
            logger.error("datasets library not installed")
            raise ImportError("Install with: pip install datasets")
        except Exception as e:
            logger.error(f"HuggingFace ingestion failed: {e}")
            raise
    
    def download_kaggle_dataset(self):
        """Download and process Kaggle dataset.
        
        Returns:
            Path to downloaded data
        """
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            if not self.config.kaggle_dataset:
                raise ValueError("kaggle_dataset is required for Kaggle source")
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            logger.info(f"Downloading Kaggle dataset: {self.config.kaggle_dataset}")
            
            # Download dataset
            download_path = self.config.cache_dir / "kaggle"
            download_path.mkdir(parents=True, exist_ok=True)
            
            if self.config.kaggle_file:
                api.dataset_download_file(
                    self.config.kaggle_dataset,
                    self.config.kaggle_file,
                    path=str(download_path)
                )
            else:
                api.dataset_download_files(
                    self.config.kaggle_dataset,
                    path=str(download_path),
                    unzip=True
                )
            
            # Extract if needed
            for zip_file in download_path.glob("*.zip"):
                logger.info(f"Extracting {zip_file}")
                with zipfile.ZipFile(zip_file, 'r') as z:
                    z.extractall(self.config.root_dir)
                logger.info(f"Extracted to {self.config.root_dir}")
            
            logger.info(f"Kaggle data saved to {self.config.root_dir}")
            return self.config.root_dir
            
        except ImportError:
            logger.error("kaggle library not installed")
            raise ImportError("Install with: pip install kaggle")
        except Exception as e:
            logger.error(f"Kaggle ingestion failed: {e}")
            raise
    
    def download_from_url(self):
        """Download and extract data from URL.
        
        Returns:
            Path to downloaded/extracted data
        """
        if not self.config.source_url:
            raise ValueError("source_url is required for URL source")
        
        # Determine filename from URL
        filename = self.config.source_url.split('/')[-1]
        if not filename or '.' not in filename:
            filename = "download.zip"
        
        local_file = self.config.cache_dir / filename
        
        # Download with progress bar
        if not local_file.exists():
            logger.info(f"Downloading from {self.config.source_url}")
            
            response = requests.get(self.config.source_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(local_file, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Downloaded to {local_file} ({get_size(local_file)})")
        else:
            logger.info(f"File already exists: {local_file} ({get_size(local_file)})")
        
        # Extract archive if it's a zip file
        if zipfile.is_zipfile(local_file):
            logger.info(f"Extracting {local_file}")
            with zipfile.ZipFile(local_file, 'r') as z:
                z.extractall(self.config.root_dir)
            logger.info(f"Extracted to {self.config.root_dir}")
        else:
            # Copy non-zip files
            import shutil
            output_file = self.config.root_dir / local_file.name
            shutil.copy2(local_file, output_file)
            logger.info(f"Copied to {output_file}")
        
        return self.config.root_dir

# ============================================================================
# src/textSummarizer/components/data_ingestion.py
# ============================================================================
# import os
# import urllib.request as request
# import zipfile
# from pathlib import Path
# from typing import Optional
# from textSummarizer.logging.logger import logger
# from textSummarizer.utils.common import get_size
# from textSummarizer.entity.config_entity import DataIngestionConfig


# class DataIngestion:
#     def __init__(self, config: DataIngestionConfig):
#         self.config = config
    
#     def download_file(self):
#         """Download file from URL if it doesn't exist"""
#         if not os.path.exists(self.config.local_data_file):
#             filename, headers = request.urlretrieve(
#                 url=self.config.source_URL,
#                 filename=self.config.local_data_file
#             )
#             logger.info(f"{filename} downloaded! with following info: \n{headers}")
#         else:
#             logger.info(f"File already exists of size: {get_size(self.config.local_data_file)}")
    
#     def extract_zip_file(self):
#         """Extract zip file into the data directory"""
#         unzip_path = self.config.unzip_dir
#         os.makedirs(unzip_path, exist_ok=True)
#         with zipfile.ZipFile(self.config.local_data_file, 'r') as zip_ref:
#             zip_ref.extractall(unzip_path)
#         logger.info(f"Extracted zip file to: {unzip_path}")
    
#     def download_huggingface_dataset(self):
#         """Download dataset from HuggingFace Hub"""
#         try:
#             from datasets import load_dataset
            
#             logger.info(f"Downloading HuggingFace dataset: {self.config.dataset_name}")
            
#             # Load dataset
#             dataset = load_dataset(
#                 self.config.dataset_name,
#                 split=self.config.dataset_split
#             )
            
#             # Save dataset to disk
#             output_path = self.config.unzip_dir / "huggingface_data"
#             os.makedirs(output_path, exist_ok=True)
            
#             # Save in different formats
#             dataset.save_to_disk(str(output_path))
#             logger.info(f"Dataset saved to: {output_path}")
            
#             # Optionally save as CSV for easy inspection
#             csv_path = output_path / f"{self.config.dataset_name}_{self.config.dataset_split}.csv"
#             dataset.to_csv(csv_path)
#             logger.info(f"Dataset also saved as CSV: {csv_path}")
            
#             # Log dataset info
#             logger.info(f"Dataset size: {len(dataset)} examples")
#             logger.info(f"Dataset features: {dataset.features}")
            
#             return dataset
            
#         except ImportError:
#             logger.error("datasets library not installed. Install with: pip install datasets")
#             raise
#         except Exception as e:
#             logger.error(f"Error downloading HuggingFace dataset: {str(e)}")
#             raise