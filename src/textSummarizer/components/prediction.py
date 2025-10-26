# ============================================================================
# src/textSummarizer/components/prediction.py
# ============================================================================
"""Prediction component for generating summaries from trained model."""

import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textSummarizer.logging.logger import logger


class PredictionPipeline:
    """Pipeline for generating summaries using trained model."""
    
    def __init__(self, model_path: str = "artifacts/model_trainer/final_model"):
        """Initialize prediction pipeline.
        
        Args:
            model_path: Path to trained model directory
        """
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model and tokenizer on initialization
        self._load_model()
    
    def _load_model(self):
        """Load trained model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            
            # Load model
            self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(
        self,
        text: str,
        max_length: int = 128,
        min_length: int = 30,
        num_beams: int = 4,
        length_penalty: float = 2.0,
        early_stopping: bool = True
    ) -> str:
        """Generate summary for input text.
        
        Args:
            text: Input article text
            max_length: Maximum summary length
            min_length: Minimum summary length
            num_beams: Beam search width
            length_penalty: Length penalty for generation
            early_stopping: Stop when all beams finish
            
        Returns:
            Generated summary text
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(
                text,
                max_length=1024,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate summary
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=min_length,
                    num_beams=num_beams,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    no_repeat_ngram_size=3
                )
            
            # Decode summary
            summary = self.tokenizer.decode(
                summary_ids[0],
                skip_special_tokens=True
            )
            
            return summary
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise


        