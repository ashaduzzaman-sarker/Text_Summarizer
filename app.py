"""FastAPI application for text summarization service."""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator
from typing import Optional
from contextlib import asynccontextmanager
import uvicorn
import time
from pathlib import Path
from textSummarizer.components.prediction import PredictionPipeline
from textSummarizer.logging.logger import logger


# Global state
predictor = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown."""
    global predictor, startup_time
    
    # Startup
    startup_time = time.time()
    try:
        logger.info("Loading prediction pipeline...")
        model_path = "artifacts/model_trainer/final_model"
        
        if not Path(model_path).exists():
            logger.error(f"Model not found at {model_path}")
            logger.error("Please train the model first by running: python main.py")
            predictor = None
        else:
            predictor = PredictionPipeline(model_path=model_path)
            logger.info(f"[SUCCESS] Prediction pipeline initialized on {predictor.device}")
            logger.info(f"Model loaded from: {model_path}")
    except Exception as e:
        logger.error(f"[FAILED] Failed to initialize pipeline: {e}")
        predictor = None
    
    yield
    
    # Shutdown
    logger.info("Shutting down API server...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Text Summarization API",
    description="Production-ready API for generating summaries from news articles using fine-tuned BART",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add compression middleware
app.add_middleware(GZipMiddleware, minimum_size=1000)


# Request/Response models
class SummarizationRequest(BaseModel):
    """Request model for summarization."""
    text: str = Field(..., min_length=50, max_length=10000, description="Article text to summarize")
    max_length: Optional[int] = Field(128, ge=30, le=512, description="Maximum summary length")
    min_length: Optional[int] = Field(30, ge=10, le=256, description="Minimum summary length")
    num_beams: Optional[int] = Field(4, ge=1, le=10, description="Beam search width")
    length_penalty: Optional[float] = Field(2.0, ge=0.5, le=3.0, description="Length penalty for generation")
    
    @field_validator('min_length')
    @classmethod
    def validate_min_max(cls, v, info):
        """Validate min_length is less than max_length."""
        if 'max_length' in info.data and v >= info.data['max_length']:
            raise ValueError('min_length must be less than max_length')
        return v


class SummarizationResponse(BaseModel):
    """Response model for summarization."""
    summary: str
    original_length: int
    summary_length: int
    compression_ratio: float
    word_count_original: int
    word_count_summary: int
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: Optional[str] = None
    uptime_seconds: Optional[float] = None
    model_path: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str
    detail: str
    timestamp: float


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            detail=str(exc),
            timestamp=time.time()
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc),
            timestamp=time.time()
        ).model_dump()
    )


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with API information."""
    model_status = "✅ Loaded" if predictor else "❌ Not Loaded"
    device = predictor.device if predictor else "N/A"
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <title>Text Summarization API</title>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                * {{ margin: 0; padding: 0; box-sizing: border-box; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                    padding: 20px;
                }}
                .container {{
                    max-width: 900px;
                    margin: 0 auto;
                    background: white;
                    border-radius: 20px;
                    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                    overflow: hidden;
                }}
                .header {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 40px;
                    text-align: center;
                }}
                .header h1 {{
                    font-size: 2.5em;
                    margin-bottom: 10px;
                }}
                .header p {{
                    font-size: 1.1em;
                    opacity: 0.9;
                }}
                .status {{
                    background: #f8f9fa;
                    padding: 20px;
                    display: flex;
                    justify-content: space-around;
                    border-bottom: 1px solid #dee2e6;
                }}
                .status-item {{
                    text-align: center;
                }}
                .status-item .label {{
                    color: #6c757d;
                    font-size: 0.9em;
                    margin-bottom: 5px;
                }}
                .status-item .value {{
                    font-size: 1.3em;
                    font-weight: bold;
                    color: #333;
                }}
                .content {{
                    padding: 40px;
                }}
                .section {{
                    margin-bottom: 30px;
                }}
                .section h2 {{
                    color: #333;
                    margin-bottom: 20px;
                    font-size: 1.8em;
                }}
                .endpoint {{
                    background: #f8f9fa;
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 10px;
                    border-left: 5px solid #667eea;
                    transition: transform 0.2s, box-shadow 0.2s;
                }}
                .endpoint:hover {{
                    transform: translateX(5px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                .endpoint strong {{
                    color: #667eea;
                    font-size: 1.1em;
                }}
                .endpoint p {{
                    color: #6c757d;
                    margin-top: 5px;
                }}
                .links {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}
                .link-card {{
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    padding: 25px;
                    border-radius: 10px;
                    text-decoration: none;
                    text-align: center;
                    transition: transform 0.2s, box-shadow 0.2s;
                    display: block;
                }}
                .link-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 10px 30px rgba(0,0,0,0.3);
                }}
                .link-card .icon {{
                    font-size: 2em;
                    margin-bottom: 10px;
                }}
                .link-card .title {{
                    font-size: 1.2em;
                    font-weight: bold;
                }}
                .footer {{
                    background: #f8f9fa;
                    padding: 20px;
                    text-align: center;
                    color: #6c757d;
                    font-size: 0.9em;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📝 Text Summarization API</h1>
                    <p>Fine-tuned BART model for news article summarization</p>
                </div>
                
                <div class="status">
                    <div class="status-item">
                        <div class="label">Model Status</div>
                        <div class="value">{model_status}</div>
                    </div>
                    <div class="status-item">
                        <div class="label">Device</div>
                        <div class="value">{device}</div>
                    </div>
                    <div class="status-item">
                        <div class="label">Version</div>
                        <div class="value">1.0.0</div>
                    </div>
                </div>
                
                <div class="content">
                    <div class="section">
                        <h2>🚀 Available Endpoints</h2>
                        
                        <div class="endpoint">
                            <strong>POST /summarize</strong>
                            <p>Generate summary for input text with customizable parameters</p>
                        </div>
                        
                        <div class="endpoint">
                            <strong>GET /health</strong>
                            <p>Check API health status and model availability</p>
                        </div>
                        
                        <div class="endpoint">
                            <strong>GET /docs</strong>
                            <p>Interactive API documentation with Swagger UI</p>
                        </div>
                        
                        <div class="endpoint">
                            <strong>GET /redoc</strong>
                            <p>Alternative API documentation with ReDoc</p>
                        </div>
                    </div>
                    
                    <div class="section">
                        <h2>🔗 Quick Links</h2>
                        <div class="links">
                            <a href="/docs" class="link-card">
                                <div class="icon">📚</div>
                                <div class="title">API Docs</div>
                            </a>
                            <a href="/health" class="link-card">
                                <div class="icon">💚</div>
                                <div class="title">Health Check</div>
                            </a>
                            <a href="/redoc" class="link-card">
                                <div class="icon">📖</div>
                                <div class="title">ReDoc</div>
                            </a>
                        </div>
                    </div>
                </div>
                
                <div class="footer">
                    Powered by FastAPI • BART Model • Transformers
                </div>
            </div>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with detailed status."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check logs and ensure model is trained."
        )
    
    uptime = time.time() - startup_time if startup_time else None
    
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        device=predictor.device,
        uptime_seconds=round(uptime, 2) if uptime else None,
        model_path=str(predictor.model_path)
    )


@app.post("/summarize", response_model=SummarizationResponse)
async def summarize(request: SummarizationRequest):
    """Generate summary for input text."""
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    start_time = time.time()
    
    try:
        logger.info(f"Summarization request: {len(request.text)} chars, beams={request.num_beams}")
        
        # Generate summary
        summary = predictor.predict(
            text=request.text,
            max_length=request.max_length,
            min_length=request.min_length,
            num_beams=request.num_beams,
            length_penalty=request.length_penalty
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        # Calculate statistics
        original_length = len(request.text)
        summary_length = len(summary)
        compression_ratio = round(summary_length / original_length * 100, 2)
        word_count_original = len(request.text.split())
        word_count_summary = len(summary.split())
        
        logger.info(f"Summarization completed in {processing_time:.2f}ms")
        
        return SummarizationResponse(
            summary=summary,
            original_length=original_length,
            summary_length=summary_length,
            compression_ratio=compression_ratio,
            word_count_original=word_count_original,
            word_count_summary=word_count_summary,
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"Summarization failed: {e}")
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
        access_log=True
    )