# ============================================================================
# src/textSummarizer/components/data_transformation.py
# ============================================================================
"""Data transformation component for tokenizing and preprocessing text."""

from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from textSummarizer.logging.logger import logger
from textSummarizer.entity.config_entity import DataTransformationConfig


class DataTransformation:
    """Handles tokenization and preprocessing of text data."""
    
    def __init__(self, config: DataTransformationConfig):
        """Initialize with configuration.
        
        Args:
            config: DataTransformationConfig instance
        """
        self.config = config
        self.tokenizer = None
    
    def load_tokenizer(self):
        """Load pretrained tokenizer.
        
        Returns:
            Loaded tokenizer
        """
        try:
            logger.info(f"Loading tokenizer: {self.config.tokenizer_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.tokenizer_name
            )
            logger.info("Tokenizer loaded successfully")
            return self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")
            raise
    
    def preprocess_function(self, examples):
        """Preprocess examples by tokenizing inputs and targets.
        
        Args:
            examples: Batch of examples from dataset
            
        Returns:
            Dictionary with tokenized inputs and labels
        """
        # Tokenize inputs (articles)
        inputs = [doc for doc in examples["article"]]
        model_inputs = self.tokenizer(
            inputs,
            max_length=self.config.max_input_length,
            padding=self.config.padding,
            truncation=True
        )
        
        # Tokenize targets (summaries/highlights)
        targets = [summary for summary in examples["highlights"]]
        labels = self.tokenizer(
            targets,
            max_length=self.config.max_target_length,
            padding=self.config.padding,
            truncation=True
        )
        
        # Add labels to model inputs
        model_inputs["labels"] = labels["input_ids"]
        
        return model_inputs
    
    def transform(self):
        """Transform dataset by tokenizing all examples.
        
        Returns:
            Transformed dataset
        """
        try:
            logger.info("Starting data transformation")
            
            # Load tokenizer
            if self.tokenizer is None:
                self.load_tokenizer()
            
            # Load dataset
            logger.info(f"Loading dataset from {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))
            logger.info(f"Dataset loaded: {len(dataset)} samples")
            
            # Apply tokenization
            logger.info("Tokenizing dataset...")
            tokenized_dataset = dataset.map(
                self.preprocess_function,
                batched=True,
                batch_size=self.config.batch_size,
                remove_columns=dataset.column_names,
                desc="Tokenizing"
            )
            
            logger.info(f"Tokenization complete: {len(tokenized_dataset)} samples")
            logger.info(f"Features: {list(tokenized_dataset.features.keys())}")
            
            # Save transformed dataset
            output_path = self.config.root_dir / "tokenized_dataset"
            tokenized_dataset.save_to_disk(str(output_path))
            logger.info(f"Transformed dataset saved to: {output_path}")
            
            # Save tokenizer for later use
            tokenizer_path = self.config.root_dir / "tokenizer"
            self.tokenizer.save_pretrained(str(tokenizer_path))
            logger.info(f"Tokenizer saved to: {tokenizer_path}")
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Data transformation failed: {e}")
            raise
