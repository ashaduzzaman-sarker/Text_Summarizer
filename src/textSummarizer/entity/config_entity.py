# ============================================================================
# src/textSummarizer/entity/config_entity.py
# ============================================================================
"""Configuration entities for the pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion stage."""
    
    root_dir: Path
    cache_dir: Path
    dataset_name: str
    config_name: str
    split: str = "train"
    max_samples: Optional[int] = None