"""
Integration tests for prediction pipeline.
Run with: pytest tests/test_integration.py -v
"""

import pytest
from pathlib import Path
from textSummarizer.components.prediction import PredictionPipeline


@pytest.fixture
def pipeline():
    """Create pipeline fixture - skips if model not found."""
    model_path = "artifacts/model_trainer/final_model"
    
    if not Path(model_path).exists():
        pytest.skip("Model not found - run training first")
    
    return PredictionPipeline(model_path=model_path)


def test_pipeline_loads(pipeline):
    """Test pipeline initializes correctly."""
    assert pipeline.model is not None
    assert pipeline.tokenizer is not None


def test_basic_prediction(pipeline):
    """Test generating a summary."""
    text = """
    The stock market rose significantly today as tech companies 
    reported strong earnings. Investors showed confidence in 
    the economic recovery as vaccination rates increase.
    """ * 3  # Make it longer
    
    summary = pipeline.predict(text)
    
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) < len(text)  # Summary should be shorter


def test_custom_parameters(pipeline):
    """Test prediction with custom generation params."""
    text = "Climate change is accelerating. " * 20
    
    summary = pipeline.predict(
        text=text,
        max_length=64,
        min_length=20,
        num_beams=2
    )
    
    assert len(summary) > 0