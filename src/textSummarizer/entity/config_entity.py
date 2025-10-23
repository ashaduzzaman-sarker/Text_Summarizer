# ============================================================================
# src/textSummarizer/entity/config_entity.py
# ============================================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    dataset_name: Optional[str] = None
    dataset_split: Optional[str] = "train"
    use_huggingface: bool = False