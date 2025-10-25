# ============================================================================
# src/textSummarizer/entity/config_entity.py
# ============================================================================
"""Configuration entities for the pipeline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Literal


@dataclass(frozen=True)
class DataIngestionConfig:
    """Configuration for data ingestion stage."""
    root_dir: Path
    cache_dir: Path
    dataset_name: str
    config_name: str
    split: str = "train"
    max_samples: Optional[int] = None


@dataclass(frozen=True)
class DataValidationConfig:
    """Configuration for data validation stage."""
    root_dir: Path
    status_file: Path
    data_dir: Path
    required_columns: List[str]
    min_samples: int


@dataclass(frozen=True)
class DataTransformationConfig:
    """Configuration for data transformation stage."""
    root_dir: Path
    data_dir: Path
    tokenizer_name: str
    max_input_length: int
    max_target_length: int
    padding: Literal["max_length", "longest", "do_not_pad"]
    batch_size: int


@dataclass(frozen=True)
class ModelTrainerConfig:
    """Configuration for model training stage."""
    root_dir: Path
    data_dir: Path
    tokenizer_dir: Path
    model_name: str
    train_split: float
    seed: int

@dataclass(frozen=True)
class ModelEvaluationConfig:
    """Configuration for model evaluation stage."""
    root_dir: Path
    data_dir: Path
    model_dir: Path
    tokenizer_dir: Path
    metric_file: Path
    report_file: Path
    predictions_file: Path