# ============================================================================
# src/textSummarizer/components/model_trainer.py
# ============================================================================
"""Model training component for fine-tuning summarization models."""

from pathlib import Path
from datasets import load_from_disk, DatasetDict
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

            import warnings
            warnings.filterwarnings('ignore', category=UserWarning, message='Moving the following attributes')
            
            logger.info("Model and tokenizer loaded successfully")
            return self.model, self.tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load model/tokenizer: {e}")
            raise
    
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
            
            logger.info(f"Training arguments configured")
            logger.info(f"Total epochs: {training_args.num_train_epochs}")
            logger.info(f"Batch size: {training_args.per_device_train_batch_size}")
            logger.info(f"Learning rate: {training_args.learning_rate}")
            
            # Data collator for dynamic padding
            data_collator = DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                model=self.model
            )
            
            # Initialize Trainer
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=self.dataset['train'],
                eval_dataset=self.dataset['validation'],
                processing_class=self.tokenizer,
                data_collator=data_collator
            )
            
            # Train
            logger.info("Training started...")
            train_result = trainer.train()
            
            # Save final model
            final_model_path = self.config.root_dir / "final_model"
            trainer.save_model(str(final_model_path))
            self.tokenizer.save_pretrained(str(final_model_path))
            
            logger.info(f"Training completed!")
            logger.info(f"Final model saved to: {final_model_path}")
            logger.info(f"Training loss: {train_result.training_loss:.4f}")
            
            # Save training metrics
            metrics_file = self.config.root_dir / "training_metrics.txt"
            with open(metrics_file, 'w') as f:
                f.write(f"Training Loss: {train_result.training_loss:.4f}\n")
                f.write(f"Training Steps: {train_result.global_step}\n")
                f.write(f"Training Time: {train_result.metrics['train_runtime']:.2f}s\n")
            
            logger.info(f"Training metrics saved to: {metrics_file}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
