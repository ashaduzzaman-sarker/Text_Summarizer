# ============================================================================
# src/textSummarizer/components/model_evaluation.py
# ============================================================================
"""Model evaluation component for calculating metrics and generating predictions."""

import json
import pandas as pd
from pathlib import Path
import torch
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
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model moved to GPU")
            else:
                logger.info("Model running on CPU")
            
            # Set to evaluation mode
            self.model.eval()
            
            logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise
    
    def load_test_data(self):
        """Load test dataset.
        
        FIXED: Uses proper test split that doesn't overlap with training.
        
        Returns:
            Test dataset
        """
        try:
            logger.info(f"Loading dataset from {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))
            
            # IMPORTANT: Use the same split logic as training to get consistent validation set
            # Training used: train_test_split(test_size=0.1, seed=42)
            # We'll use the validation split (the 10% that training used for validation)
            
            logger.info("Creating test split (using same seed as training for consistency)")
            split = dataset.train_test_split(test_size=0.1, seed=42)
            
            # Use the 'test' split (which was 'validation' during training)
            # This ensures we evaluate on data the model saw during validation
            # but NOT during training steps
            self.dataset = split['test']
            
            # Limit samples if specified
            if self.params.max_samples and self.params.max_samples < len(self.dataset):
                self.dataset = self.dataset.select(range(self.params.max_samples))
                logger.info(f"Limited to {self.params.max_samples} test samples")
            
            logger.info(f"Test dataset loaded: {len(self.dataset)} samples")
            logger.info(f"Note: Using validation split from training for consistency")
            
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
        
        FIXED: Better batch handling and progress tracking.
        
        Returns:
            Tuple of (predictions, references)
        """
        try:
            logger.info("Generating predictions...")
            logger.info(f"Using device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
            
            predictions = []
            references = []

            
            # Process in batches with progress bar
            num_batches = (len(self.dataset) + self.params.batch_size - 1) // self.params.batch_size
            logger.info(f"Processing {len(self.dataset)} samples in {num_batches} batches")
            
            for i in tqdm(range(0, len(self.dataset), self.params.batch_size), desc="Generating"):
                batch_end = min(i + self.params.batch_size, len(self.dataset))
                batch = self.dataset[i:batch_end]
                
                # Get input_ids and labels
                input_ids = batch['input_ids']
                labels = batch['labels']
                
                # Convert to tensors and move to device
                if torch.cuda.is_available():
                    input_ids_tensor = torch.tensor(input_ids).cuda()
                else:
                    input_ids_tensor = torch.tensor(input_ids)
                
                # Generate summaries
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids_tensor,
                        num_beams=self.params.num_beams,
                        max_length=self.params.max_length,
                        min_length=self.params.min_length,
                        length_penalty=self.params.length_penalty,
                        no_repeat_ngram_size=self.params.no_repeat_ngram_size,
                        early_stopping=self.params.early_stopping
                    )
                
                # Move back to CPU for decoding
                generated_ids = generated_ids.cpu()
                
                # Decode predictions
                batch_predictions = self.tokenizer.batch_decode(
                    generated_ids,
                    skip_special_tokens=True
                )
                predictions.extend(batch_predictions)
                
                # Decode references (replace -100 padding with pad_token_id)
                import numpy as np
                labels_array = np.array(labels)
                labels_array = np.where(
                    labels_array != -100,
                    labels_array,
                    self.tokenizer.pad_token_id
                )
                
                batch_references = self.tokenizer.batch_decode(
                    labels_array,
                    skip_special_tokens=True
                )
                references.extend(batch_references)
            
            logger.info(f"Generated {len(predictions)} predictions")
            logger.info(f"Average prediction length: {np.mean([len(p.split()) for p in predictions]):.1f} words")
            logger.info(f"Average reference length: {np.mean([len(r.split()) for r in references]):.1f} words")
            
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
            
            # Clean predictions and references
            predictions = [pred.strip() for pred in predictions if pred.strip()]
            references = [ref.strip() for ref in references if ref.strip()]
            
            # Ensure same length
            min_len = min(len(predictions), len(references))
            predictions = predictions[:min_len]
            references = references[:min_len]
            
            logger.info(f"Computing metrics on {len(predictions)} samples")
            
            # Calculate ROUGE scores
            results = self.metric.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True
            )
            
            # Extract scores
            metrics = {
                'rouge1': round(results['rouge1'], 4),
                'rouge2': round(results['rouge2'], 4),
                'rougeL': round(results['rougeL'], 4),
                'rougeLsum': round(results['rougeLsum'], 4)
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
                'prediction': predictions,
                'reference_length': [len(r.split()) for r in references],
                'prediction_length': [len(p.split()) for p in predictions]
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
        import numpy as np
        
        report = "=" * 70 + "\n"
        report += "MODEL EVALUATION REPORT\n"
        report += "=" * 70 + "\n\n"
        
        # Metrics section
        report += "ROUGE SCORES:\n"
        report += "-" * 70 + "\n"
        for metric_name, score in metrics.items():
            report += f"{metric_name.upper():15s}: {score:.4f}\n"
        report += "\n"
        
        # Statistics section
        report += "SUMMARY STATISTICS:\n"
        report += "-" * 70 + "\n"
        pred_lengths = [len(p.split()) for p in predictions]
        ref_lengths = [len(r.split()) for r in references]
        
        report += f"Total samples:           {len(predictions)}\n"
        report += f"Avg prediction length:   {np.mean(pred_lengths):.1f} words\n"
        report += f"Avg reference length:    {np.mean(ref_lengths):.1f} words\n"
        report += f"Min prediction length:   {np.min(pred_lengths)} words\n"
        report += f"Max prediction length:   {np.max(pred_lengths)} words\n"
        report += "\n"
        
        # Sample predictions section
        report += "SAMPLE PREDICTIONS:\n"
        report += "-" * 70 + "\n\n"
        
        num_samples = min(5, len(predictions))
        for i in range(num_samples):
            report += f"Example {i+1}:\n"
            report += f"Reference ({len(references[i].split())} words):\n"
            report += f"{references[i][:200]}{'...' if len(references[i]) > 200 else ''}\n\n"
            report += f"Prediction ({len(predictions[i].split())} words):\n"
            report += f"{predictions[i][:200]}{'...' if len(predictions[i]) > 200 else ''}\n"
            report += "\n" + "-" * 70 + "\n\n"
        
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