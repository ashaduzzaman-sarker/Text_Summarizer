# ============================================================================
# src/textSummarizer/components/model_trainer.py
# ============================================================================
"""Model training component for fine-tuning summarization models."""

import numpy as np
from pathlib import Path
from datasets import load_from_disk, DatasetDict
from transformers.trainer_callback import EarlyStoppingCallback
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from textSummarizer.logging.logger import logger
from textSummarizer.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    """Handles model training and fine-tuning."""
    
    def __init__(self, config: ModelTrainerConfig, params: dict):
        """Initialize with configuration.
        
        Args:
            config: ModelTrainerConfig instance
            params: Training parameters from params.yaml
        """
        self.config = config
        self.params = params
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.metric = None
    
    def load_data(self):
        """Load tokenized dataset and split into train/validation.
        
        Returns:
            DatasetDict with train and validation splits
        """
        try:
            logger.info(f"Loading tokenized dataset from {self.config.data_dir}")
            dataset = load_from_disk(str(self.config.data_dir))
            logger.info(f"Dataset loaded: {len(dataset)} samples")
            
            # Split into train and validation
            logger.info(f"Splitting dataset: {self.config.train_split:.0%} train")
            split_dataset = dataset.train_test_split(
                test_size=1 - self.config.train_split,
                seed=self.config.seed
            )
            
            # Rename 'test' to 'validation' for clarity
            self.dataset = DatasetDict({
                'train': split_dataset['train'],
                'validation': split_dataset['test']
            })
            
            logger.info(f"Train samples: {len(self.dataset['train'])}")
            logger.info(f"Validation samples: {len(self.dataset['validation'])}")
            
            return self.dataset
            
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def load_model_and_tokenizer(self):
        """Load pretrained model and tokenizer.
        
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # Load tokenizer (saved from transformation stage)
            logger.info(f"Loading tokenizer from {self.config.tokenizer_dir}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.config.tokenizer_dir)
            )
            
            # Load pretrained model
            logger.info(f"Loading model: {self.config.model_name}")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.config.model_name
            )
            
            # Suppress generation config warnings
            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, message='Moving the following attributes')
            
            logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise
    
    def load_metric(self):
        """Load ROUGE metric for evaluation during training.
        
        Returns:
            ROUGE metric object
        """
        try:
            import evaluate
            self.metric = evaluate.load("rouge")
            logger.info("ROUGE metric loaded for training evaluation")
            return self.metric
            
        except ImportError:
            logger.error("evaluate library not installed")
            raise ImportError("Install with: pip install evaluate rouge-score")
        except Exception as e:
            logger.error(f"Failed to load ROUGE metric: {e}")
            raise
    
    def compute_metrics(self, eval_pred):
        """Compute ROUGE metrics during evaluation.
        
        This function is called by the Trainer during evaluation to compute
        metrics on the validation set.
        
        Args:
            eval_pred: EvalPrediction object containing predictions and labels
            
        Returns:
            Dictionary of computed metrics
        """
        try:
            predictions, labels = eval_pred
            
            # Decode predictions (generated token IDs)
            if isinstance(predictions, tuple):
                predictions = predictions[0]
            
            # Handle different prediction shapes
            if len(predictions.shape) == 3:
                # If predictions are logits, get the argmax
                predictions = np.argmax(predictions, axis=-1)
            
            # Decode predictions to text
            decoded_preds = self.tokenizer.batch_decode(
                predictions,
                skip_special_tokens=True
            )
            
            # Replace -100 in labels (used for padding in loss calculation)
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
            
            # Decode labels to text
            decoded_labels = self.tokenizer.batch_decode(
                labels,
                skip_special_tokens=True
            )
            
            # Clean up whitespace
            decoded_preds = [pred.strip() for pred in decoded_preds]
            decoded_labels = [label.strip() for label in decoded_labels]
            
            # Compute ROUGE scores
            result = self.metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                use_stemmer=True
            )
            
            # Extract and round metrics
            metrics = {
                "rouge1": round(result["rouge1"], 4),
                "rouge2": round(result["rouge2"], 4),
                "rougeL": round(result["rougeL"], 4),
                "rougeLsum": round(result["rougeLsum"], 4)
            }
            
            # Log metrics for visibility
            logger.info(f"Validation metrics: {metrics}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to compute metrics: {e}")
            # Return empty dict to avoid breaking training
            return {}
    
    def train(self):
        """Train the model using HuggingFace Trainer.
        
        Returns:
            Training output
        """
        try:
            logger.info("Starting model training")
            
            # Load data
            if self.dataset is None:
                self.load_data()
            
            # Load model and tokenizer
            if self.model is None or self.tokenizer is None:
                self.load_model_and_tokenizer()
            
            # Load ROUGE metric
            if self.metric is None:
                self.load_metric()
            
            # Setup training arguments
            training_args = Seq2SeqTrainingArguments(
                output_dir=self.params.output_dir,
                num_train_epochs=self.params.num_train_epochs,
                per_device_train_batch_size=self.params.per_device_train_batch_size,
                per_device_eval_batch_size=self.params.per_device_eval_batch_size,
                warmup_steps=self.params.warmup_steps,
                weight_decay=self.params.weight_decay,
                logging_dir=f"{self.config.root_dir}/logs",
                logging_steps=self.params.logging_steps,
                eval_strategy=self.params.eval_strategy,
                eval_steps=self.params.eval_steps,
                save_steps=self.params.save_steps,
                save_total_limit=self.params.save_total_limit,
                learning_rate=self.params.learning_rate,
                gradient_accumulation_steps=self.params.gradient_accumulation_steps,
                fp16=self.params.fp16,
                predict_with_generate=self.params.predict_with_generate,
                generation_max_length=self.params.generation_max_length,
                load_best_model_at_end=self.params.load_best_model_at_end,
                metric_for_best_model=self.params.metric_for_best_model,
                greater_is_better=self.params.greater_is_better,
                report_to=["tensorboard"],
                seed=self.config.seed
            )
            
            logger.info("Training arguments configured")
            logger.info(f"Total epochs: {training_args.num_train_epochs}")
            logger.info(f"Batch size (train/eval): {training_args.per_device_train_batch_size}/{training_args.per_device_eval_batch_size}")
            logger.info(f"Learning rate: {training_args.learning_rate}")
            logger.info(f"Best model metric: {self.params.metric_for_best_model}")
            
            # Data collator for dynamic padding
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model
            )
            

            # Initialize Trainer with compute_metrics and early stopping
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['validation'],
                processing_class=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]  # Stop if no improvement for 3 evals
            )
            
            # Train
            logger.info("Training started...")
            logger.info("ROUGE metrics will be computed during evaluation")
            train_result = trainer.train()
            
            # Save final model
            final_model_path = self.config.root_dir / "final_model"
            trainer.save_model(str(final_model_path))
            self.tokenizer.save_pretrained(str(final_model_path))
            
            logger.info("Training completed!")
            logger.info(f"Final model saved to: {final_model_path}")
            logger.info(f"Training loss: {train_result.training_loss:.4f}")
            
            # Get final evaluation metrics
            logger.info("Running final evaluation...")
            final_metrics = trainer.evaluate()
            
            # Save comprehensive training metrics
            metrics_file = self.config.root_dir / "training_metrics.txt"
            with open(metrics_file, 'w') as f:
                f.write("=" * 70 + "\n")
                f.write("TRAINING METRICS\n")
                f.write("=" * 70 + "\n\n")
                
                f.write("Training Summary:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Training Loss: {train_result.training_loss:.4f}\n")
                f.write(f"Training Steps: {train_result.global_step}\n")
                f.write(f"Training Time: {train_result.metrics['train_runtime']:.2f}s\n")
                f.write(f"Samples per Second: {train_result.metrics['train_samples_per_second']:.2f}\n")
                f.write("\n")
                
                f.write("Final Validation Metrics:\n")
                f.write("-" * 70 + "\n")
                f.write(f"Validation Loss: {final_metrics.get('eval_loss', 'N/A'):.4f}\n")
                f.write(f"ROUGE-1: {final_metrics.get('eval_rouge1', 'N/A'):.4f}\n")
                f.write(f"ROUGE-2: {final_metrics.get('eval_rouge2', 'N/A'):.4f}\n")
                f.write(f"ROUGE-L: {final_metrics.get('eval_rougeL', 'N/A'):.4f}\n")
                f.write(f"ROUGE-Lsum: {final_metrics.get('eval_rougeLsum', 'N/A'):.4f}\n")
                f.write("\n")
                f.write("=" * 70 + "\n")
            
            logger.info(f"Training metrics saved to: {metrics_file}")
            logger.info("Final validation metrics:")
            logger.info(f"  Loss: {final_metrics.get('eval_loss', 'N/A'):.4f}")
            logger.info(f"  ROUGE-1: {final_metrics.get('eval_rouge1', 'N/A'):.4f}")
            logger.info(f"  ROUGE-2: {final_metrics.get('eval_rouge2', 'N/A'):.4f}")
            logger.info(f"  ROUGE-L: {final_metrics.get('eval_rougeL', 'N/A'):.4f}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise