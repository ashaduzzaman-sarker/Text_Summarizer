# ============================================================================
# README_DEPLOYMENT.md
# ============================================================================
# Text Summarizer Deployment Guide

## Quick Start

### Option 1: FastAPI (Production)

1. Install dependencies:
```bash
pip install fastapi uvicorn pydantic python-multipart
```

2. Run API:
```bash
python app.py
```

3. Access:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

4. Test:
```bash
python test_api.py
```

### Option 2: Gradio (Demo)

1. Install dependencies:
```bash
pip install gradio
```

2. Run demo:
```bash
python gradio_app.py
```

3. Access:
- Demo: http://localhost:7860

### Option 3: Docker

1. Build image:
```bash
docker build -t text-summarizer .
```

2. Run FastAPI:
```bash
docker run -p 8000:8000 -v $(pwd)/artifacts:/app/artifacts text-summarizer python app.py
```

3. Run Gradio:
```bash
docker run -p 7860:7860 -v $(pwd)/artifacts:/app/artifacts text-summarizer python gradio_app.py
```

### Option 4: Docker Compose (Both services)

1. Start both services:
```bash
docker-compose up -d
```

2. Access:
- API: http://localhost:8000
- Demo: http://localhost:7860

3. Stop:
```bash
docker-compose down
```

## API Usage

### Python
```python
import requests

response = requests.post(
    "http://localhost:8000/summarize",
    json={
        "text": "Your article text here...",
        "max_length": 128,
        "min_length": 30
    }
)

summary = response.json()["summary"]
```

### cURL
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{"text": "Your article text..."}'
```

### JavaScript
```javascript
fetch('http://localhost:8000/summarize', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Your article text...',
    max_length: 128
  })
})
.then(res => res.json())
.then(data => console.log(data.summary));
```

## Production Deployment

### Environment Variables
```bash
export MODEL_PATH=artifacts/model_trainer/final_model
export PORT=8000
export WORKERS=4
```

### Run with Gunicorn
```bash
pip install gunicorn
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Nginx Configuration
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

## Monitoring

### Logging
Logs are saved to: `logs/running_logs.log`

### Metrics
- Request count
- Response time
- Error rate
- Model inference time

## Troubleshooting

### Model not found
- Ensure model exists: `artifacts/model_trainer/final_model`
- Check MODEL_PATH environment variable

### Out of memory
- Reduce batch size
- Use CPU instead of GPU
- Implement request queuing

### Slow inference
- Use GPU if available
- Reduce num_beams
- Implement model caching