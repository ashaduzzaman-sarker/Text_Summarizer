# ============================================================================
# src/textSummarizer/components/model_evaluation.py
# ============================================================================
"""Model evaluation component for calculating metrics and generating predictions."""

import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from textSummarizer.logging.logger import logger
from textSummarizer.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    """Handles model evaluation and metrics calculation."""
    
    def __init__(self, config: ModelEvaluationConfig, params: dict):
        """Initialize with configuration.
        
        Args:
            config: ModelEvaluationConfig instance
            params: Evaluation parameters from params.yaml
        """
        self.config = config
        self.params = params
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.metric = None
    
    def load_model_and_tokenizer(self):
        """Load trained model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            logger.info(f"Loading tokenizer from {self.config.tokenizer_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.tokenizer_dir)
            )
            
            logger.info(f"Loading model from {self.config.model_dir}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                str(self.config.model_dir)
            )
            
            # Move model to GPU if available
            import torch
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model moved to GPU")
            else:
                logger.info("Model running on CPU")
            
            logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise
    
    def load_test_data(self):
        """Load test dataset.
        
        Returns:
            Test dataset
        """
        try:
            logger.info(f"Loading dataset from {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))
            
            # Use validation split (10% from training stage split)
            # Or load full dataset and create test split
            if hasattr(dataset, 'train_test_split'):
                # If it's a single dataset, split it
                split = dataset.train_test_split(test_size=0.1, seed=42)
                self.dataset = split['test']
            else:
                # Use the full dataset as test
                self.dataset = dataset
            
            # Limit samples if specified
            if self.params.max_samples and self.params.max_samples < len(self.dataset):
                self.dataset = self.dataset.select(range(self.params.max_samples))
                logger.info(f"Limited to {self.params.max_samples} test samples")
            
            logger.info(f"Test dataset loaded: {len(self.dataset)} samples")
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load test data: {e}")
            raise
    
    def load_rouge_metric(self):
        """Load ROUGE metric.
        
        Returns:
            ROUGE metric object
        """
        try:
            import evaluate
            self.metric = evaluate.load("rouge")
            logger.info("ROUGE metric loaded")
            return self.metric
            
        except ImportError:
            logger.error("evaluate library not installed")
            raise ImportError("Install with: pip install evaluate rouge-score")
        except Exception as e:
            logger.error(f"Failed to load ROUGE metric: {e}")
            raise
    
    def generate_predictions(self):
        """Generate predictions for test dataset.
        
        Returns:
            Tuple of (predictions, references)
        """
        try:
            logger.info("Generating predictions...")
            
            predictions = []
            references = []
            
            # Process in batches
            for i in tqdm(range(0, len(self.dataset), self.params.batch_size)):
                batch = self.dataset[i:i + self.params.batch_size]
                
                # Get input_ids and labels
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # Generate summaries
                import torch
                with torch.no_grad():
                    if torch.cuda.is_available():
                        input_ids = torch.tensor(input_ids).cuda()
                    else:
                        input_ids = torch.tensor(input_ids)
                    
                    generated_ids = self.model.generate(
                        input_ids,
                        num_beams=self.params.num_beams,
                        max_length=self.params.max_length,
                        min_length=self.params.min_length,
                        length_penalty=self.params.length_penalty,
                        no_repeat_ngram_size=self.params.no_repeat_ngram_size,
                        early_stopping=self.params.early_stopping
                    )
                
                # Decode predictions
                batch_predictions = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                predictions.extend(batch_predictions)
                
                # Decode references
                batch_references = self.tokenizer.batch_decode(
                    labels,
                    skip_special_tokens=True
                )
                references.extend(batch_references)
            
            logger.info(f"Generated {len(predictions)} predictions")
            return predictions, references
            
        except Exception as e:
            logger.error(f"Failed to generate predictions: {e}")
            raise
    
    def calculate_metrics(self, predictions, references):
        """Calculate ROUGE metrics.
        
        Args:
            predictions: List of predicted summaries
            references: List of reference summaries
            
        Returns:
            Dictionary of metrics
        """
        try:
            logger.info("Calculating ROUGE metrics...")
            
            # Calculate ROUGE scores
            results = self.metric.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )
            
            # Extract scores
            metrics = {
                'rouge1': results['rouge1'],
                'rouge2': results['rouge2'],
                'rougeL': results['rougeL'],
                'rougeLsum': results['rougeLsum']
            }
            
            logger.info("Metrics calculated:")
            for metric_name, score in metrics.items():
                logger.info(f"  {metric_name}: {score:.4f}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to calculate metrics: {e}")
            raise
    
    def save_results(self, metrics, predictions, references):
        """Save evaluation results to files.
        
        Args:
            metrics: Dictionary of calculated metrics
            predictions: List of predicted summaries
            references: List of reference summaries
        """
        try:
            # Save metrics as JSON
            with open(self.config.metric_file, 'w') as f:
                json.dump(metrics, f, indent=4)
            logger.info(f"Metrics saved to {self.config.metric_file}")
            
            # Save predictions as CSV
            df = pd.DataFrame({
                'reference': references,
                'prediction': predictions
            })
            df.to_csv(self.config.predictions_file, index=False)
            logger.info(f"Predictions saved to {self.config.predictions_file}")
            
            # Create evaluation report
            report = self._create_report(metrics, predictions, references)
            self.config.report_file.write_text(report)
            logger.info(f"Report saved to {self.config.report_file}")
            
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            raise
    
    def _create_report(self, metrics, predictions, references):
        """Create human-readable evaluation report.
        
        Args:
            metrics: Dictionary of metrics
            predictions: List of predictions
            references: List of references
            
        Returns:
            Report string
        """
        report = "=" * 70 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Metrics section
        report += "ROUGE SCORES:\n"
        report += "-" * 70 + "\n"
        for metric_name, score in metrics.items():
            report += f"{metric_name.upper():15s}: {score:.4f}\n"
        report += "\n"
        
        # Sample predictions section
        report += "SAMPLE PREDICTIONS:\n"
        report += "-" * 70 + "\n\n"
        
        num_samples = min(5, len(predictions))
        for i in range(num_samples):
            report += f"Example {i+1}:\n"
            report += f"Reference: {references[i][:200]}...\n"
            report += f"Prediction: {predictions[i][:200]}...\n"
            report += "\n"
        
        report += "=" * 70 + "\n"
        
        return report
    
    def evaluate(self):
        """Run complete evaluation pipeline.
        
        Returns:
            Dictionary of metrics
        """
        try:
            logger.info("Starting model evaluation")
            
            # Load model and tokenizer
            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()
            
            # Load test data
            if self.dataset is None:
                self.load_test_data()
            
            # Load metric
            if self.metric is None:
                self.load_rouge_metric()
            
            # Generate predictions
            predictions, references = self.generate_predictions()
            
            # Calculate metrics
            metrics = self.calculate_metrics(predictions, references)
            
            # Save results
            self.save_results(metrics, predictions, references)
            
            logger.info("Evaluation completed successfully")
            return metrics
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            raise



