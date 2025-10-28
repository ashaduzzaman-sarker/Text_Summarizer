# ============================================================================
# app.py (FastAPI Application)
# ============================================================================
"""FastAPI application for text summarization service."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.logging.logger import logger


# Initialize FastAPI app
app = FastAPI(
    title="Text Summarization API",
    description="API for generating summaries from news articles using fine-tuned BART",
    version="1.0.0"
)

# Initialize prediction pipeline (load model once at startup)
try:
    predictor = PredictionPipeline()
    logger.info("Prediction pipeline initialized")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {e}")
    predictor = None


# Request/Response models
class SummarizationRequest(BaseModel):
    """Request model for summarization."""
    text: str = Field(..., min_length=50, description="Article text to summarize")
    max_length: Optional[int] = Field(128, ge=30, le=512, description="Maximum summary length")
    min_length: Optional[int] = Field(30, ge=10, le=256, description="Minimum summary length")
    num_beams: Optional[int] = Field(4, ge=1, le=10, description="Beam search width")


class SummarizationResponse(BaseModel):
    """Response model for summarization."""
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    html_content = """
    <!DOCTYPE html>
    <html>
        <head>
            <title>Text Summarization API</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 50px auto;
                    padding: 20px;
                    background: #f5f5f5;
                }
                .container {
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }
                h1 { color: #333; }
                .endpoint {
                    background: #f8f9fa;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border-left: 4px solid #007bff;
                }
                a { color: #007bff; text-decoration: none; }
                a:hover { text-decoration: underline; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>📝 Text Summarization API</h1>
                <p>Fine-tuned BART model for news article summarization</p>
                
                <h2>Available Endpoints:</h2>
                
                <div class="endpoint">
                    <strong>POST /summarize</strong><br>
                    Generate summary for input text
                </div>
                
                <div class="endpoint">
                    <strong>GET /health</strong><br>
                    Check API health status
                </div>
                
                <div class="endpoint">
                    <strong>GET /docs</strong><br>
                    Interactive API documentation (Swagger UI)
                </div>
                
                <div class="endpoint">
                    <strong>GET /redoc</strong><br>
                    Alternative API documentation (ReDoc)
                </div>
                
                <h2>Quick Links:</h2>
                <ul>
                    <li><a href="/docs">📚 API Documentation</a></li>
                    <li><a href="/health">💚 Health Check</a></li>
                </ul>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "status": "healthy",
        "model_loaded": True,
        "device": predictor.device
    }


@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    """Generate summary for input text.
    
    Args:
        request: SummarizationRequest with text and parameters
        
    Returns:
        SummarizationResponse with summary and statistics
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Generate summary
        summary = predictor.predict(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams
        )
        
        # Calculate statistics
        original_length = len(request.text)
        summary_length = len(summary)
        compression_ratio = round(summary_length / original_length * 100, 2)
        
        return SummarizationResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio
        )
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Run application
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )