# ============================================================================
# src/textSummarizer/components/data_validation.py
# ============================================================================
"""Data validation component for checking dataset quality."""

from pathlib import Path
from datasets import load_from_disk, Dataset
from textSummarizer.logging.logger import logger
from textSummarizer.entity.config_entity import DataValidationConfig


class DataValidation:
    """Validates dataset structure and quality."""
    
    def __init__(self, config: DataValidationConfig):
        """Initialize with configuration.
        
        Args:
            config: DataValidationConfig instance
        """
        self.config = config
        self.validation_status = True
        self.validation_errors = []
    
    def validate_dataset_exists(self) -> bool:
        """Check if dataset directory exists.
        
        Returns:
            True if dataset exists, False otherwise
        """
        if not self.config.data_dir.exists():
            error_msg = f"Dataset directory not found: {self.config.data_dir}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False
        
        logger.info(f"Dataset directory found: {self.config.data_dir}")
        return True
    
    def validate_columns(self, dataset: Dataset) -> bool:
        """Validate required columns exist in dataset.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if all required columns exist, False otherwise
        """
        dataset_columns = list(dataset.features.keys())
        logger.info(f"Dataset columns: {dataset_columns}")
        
        missing_columns = [
            col for col in self.config.required_columns 
            if col not in dataset_columns
        ]
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            logger.error(error_msg)
            self.validation_errors.append(error_msg)
            return False
        
        logger.info("All required columns present")
        return True
    
    def validate_data_quality(self, dataset: Dataset) -> bool:
        """Validate data quality checks.
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            True if quality checks pass, False otherwise
        """
        quality_passed = True
        
        # Check minimum samples
        num_samples = len(dataset)
        logger.info(f"Dataset size: {num_samples} samples")
        
        if num_samples < self.config.min_samples:
            error_msg = (
                f"Dataset too small: {num_samples} samples "
                f"(minimum: {self.config.min_samples})"
            )
            logger.warning(error_msg)
            self.validation_errors.append(error_msg)
            quality_passed = False
        
        # Check for null values in required columns
        for column in self.config.required_columns:
            null_count = sum(1 for item in dataset[column] if not item)
            
            if null_count > 0:
                null_percentage = (null_count / num_samples) * 100
                warning_msg = (
                    f"Column '{column}' has {null_count} null values "
                    f"({null_percentage:.2f}%)"
                )
                logger.warning(warning_msg)
                self.validation_errors.append(warning_msg)
                
                if null_percentage > 5.0:  # Fail if >5% null
                    quality_passed = False
        
        if quality_passed:
            logger.info("Data quality checks passed")
        
        return quality_passed
    
    def validate(self) -> bool:
        """Run all validation checks.
        
        Returns:
            True if all validations pass, False otherwise
        """
        try:
            logger.info("Starting data validation")
            
            # Check if dataset exists
            if not self.validate_dataset_exists():
                self.validation_status = False
                self._write_status()
                return False
            
            # Load dataset
            logger.info(f"Loading dataset from {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))
            
            # Validate columns
            if not self.validate_columns(dataset):
                self.validation_status = False
            
            # Validate data quality
            if not self.validate_data_quality(dataset):
                self.validation_status = False
            
            # Write validation status
            self._write_status()
            
            if self.validation_status:
                logger.info("All validation checks passed")
            else:
                logger.error(f"Validation failed with {len(self.validation_errors)} errors")
            
            return self.validation_status
            
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            self.validation_status = False
            self.validation_errors.append(str(e))
            self._write_status()
            raise
    
    def _write_status(self) -> None:
        """Write validation status to file."""
        status_content = "Validation Status: "
        status_content += "PASSED" if self.validation_status else "FAILED"
        
        if self.validation_errors:
            status_content += "\n\nErrors/Warnings:\n"
            status_content += "\n".join(f"- {error}" for error in self.validation_errors)
        
        self.config.status_file.write_text(status_content)
        logger.info(f"Validation status written to {self.config.status_file}")
