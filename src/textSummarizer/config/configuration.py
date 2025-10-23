# ============================================================================
# src/textSummarizer/entity/config_entity.py
# ============================================================================
from textSummarizer.constants.constants import *
from textSummarizer.utils.common import read_yaml, create_directories
from textSummarizer.entity.config_entity import DataIngestionConfig
class ConfigurationManager:
    def __init__(
        self,
        config_filepath: Path = CONFIG_FILE_PATH,
        params_filepath: Path = PARAMS_FILE_PATH
    ):
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)
        create_directories([self.config.artifacts_root])
    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion
        create_directories([config.root_dir])
        
        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_URL=config.source_URL,
            local_data_file=Path(config.local_data_file),
            unzip_dir=Path(config.unzip_dir),
            dataset_name=config.get('dataset_name'),
            dataset_split=config.get('dataset_split', 'train'),
            use_huggingface=config.get('use_huggingface', False)
        )
        
        return data_ingestion_config