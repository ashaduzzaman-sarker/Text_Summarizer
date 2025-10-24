from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Literal


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion from multiple sources."""
    
    root_dir: Path
    source_type: Literal["huggingface", "kaggle", "url"]
    
    # HuggingFace
    hf_dataset_name: Optional[str] = None
    hf_config_name: Optional[str] = None
    hf_split: str = "train"
    hf_streaming: bool = False
    
    # Kaggle
    kaggle_dataset: Optional[str] = None
    kaggle_file: Optional[str] = None
    
    # URL
    source_url: Optional[str] = None
    
    # Common
    output_format: Literal["arrow", "parquet", "csv", "json"] = "arrow"
    cache_dir: Path = Path("artifacts/data_ingestion/cache")
    max_samples: Optional[int] = None

# ============================================================================
# src/textSummarizer/entity/config_entity.py
# ============================================================================
# from dataclasses import dataclass
# from pathlib import Path
# from typing import Optional


# @dataclass(frozen=True)
# class DataIngestionConfig:
#     root_dir: Path
#     source_URL: str
#     local_data_file: Path
#     unzip_dir: Path
#     dataset_name: Optional[str] = None
#     dataset_split: Optional[str] = "train"
#     use_huggingface: bool = False