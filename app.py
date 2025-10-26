# ============================================================================
# app.py (FastAPI Application)
# ============================================================================
"""FastAPI application for text summarization."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from src.textSummarizer.components.prediction import PredictionPipeline
from src.textSummarizer.logging.logger import logger

# Initialize FastAPI app
app = FastAPI(
    title="Text Summarizer API",
    description="API for generating summaries using fine-tuned BART model",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize prediction pipeline (loaded once at startup)
predictor = None


class SummarizeRequest(BaseModel):
    """Request model for summarization."""
    text: str = Field(..., description="Article text to summarize", min_length=50)
    max_length: Optional[int] = Field(128, description="Maximum summary length", ge=30, le=512)
    min_length: Optional[int] = Field(30, description="Minimum summary length", ge=10, le=256)
    num_beams: Optional[int] = Field(4, description="Beam search width", ge=1, le=10)
    length_penalty: Optional[float] = Field(2.0, description="Length penalty", ge=0.5, le=5.0)


class SummarizeResponse(BaseModel):
    """Response model for summarization."""
    summary: str
    input_length: int
    summary_length: int


@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    global predictor
    try:
        logger.info("Loading model...")
        predictor = PredictionPipeline()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Text Summarizer API",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "summarize": "/summarize",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": predictor is not None
    }


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: SummarizeRequest):
    """Generate summary for input text.
    
    Args:
        request: Summarization request
        
    Returns:
        Generated summary with metadata
    """
    try:
        if predictor is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Generate summary
        summary = predictor.predict(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty
        )
        
        return SummarizeResponse(
            summary=summary,
            input_length=len(request.text.split()),
            summary_length=len(summary.split())
        )
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)