from pathlib import Path
from textSummarizer.constants.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity.config_entity import DataIngestionConfig


class ConfigurationManager:
    """Manages all configuration loading and validation."""
    
    def __init__(
        self,
        config_path: Path = CONFIG_FILE_PATH,
        params_path: Path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_path)
        self.params = read_yaml(params_path)
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """Get data ingestion configuration.
        
        Returns:
            DataIngestionConfig instance
        """
        cfg = self.config.data_ingestion
        create_directories([cfg.root_dir])
        
        return DataIngestionConfig(
            root_dir=Path(cfg.root_dir),
            source_type=cfg.source_type,
            hf_dataset_name=cfg.get('hf_dataset_name'),
            hf_config_name=cfg.get('hf_config_name'),
            hf_split=cfg.get('hf_split', 'train'),
            hf_streaming=cfg.get('hf_streaming', False),
            kaggle_dataset=cfg.get('kaggle_dataset'),
            kaggle_file=cfg.get('kaggle_file'),
            source_url=cfg.get('source_url'),
            output_format=cfg.get('output_format', 'arrow'),
            cache_dir=Path(cfg.get('cache_dir', 'artifacts/data_ingestion/cache')),
            max_samples=cfg.get('max_samples')
        )
# ============================================================================
# src/textSummarizer/entity/config_entity.py
# ============================================================================
# from textSummarizer.constants.constants import *
# from textSummarizer.utils.common import read_yaml, create_directories
# from textSummarizer.entity.config_entity import DataIngestionConfig
# class ConfigurationManager:
#     def __init__(
#         self,
#         config_filepath: Path = CONFIG_FILE_PATH,
#         params_filepath: Path = PARAMS_FILE_PATH
#     ):
#         self.config = read_yaml(config_filepath)
#         self.params = read_yaml(params_filepath)
#         create_directories([self.config.artifacts_root])
    
#     def get_data_ingestion_config(self) -> DataIngestionConfig:
#         config = self.config.data_ingestion
#         create_directories([config.root_dir])
        
#         data_ingestion_config = DataIngestionConfig(
#             root_dir=Path(config.root_dir),
#             source_URL=config.source_URL,
#             local_data_file=Path(config.local_data_file),
#             unzip_dir=Path(config.unzip_dir),
#             dataset_name=config.get('dataset_name'),
#             dataset_split=config.get('dataset_split', 'train'),
#             use_huggingface=config.get('use_huggingface', False)
#         )
        
#         return data_ingestion_config