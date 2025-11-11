# 📝 Text Summarizer - End-to-End NLP Project

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-purple.svg)

An end-to-end production-ready text summarization system built with **BART** (Bidirectional and Auto-Regressive Transformers), fine-tuned on the **CNN/DailyMail** dataset. Features a complete MLOps pipeline with FastAPI REST API and Gradio web interface.

---

## 🌟 Features

- ✅ **State-of-the-art Model**: Fine-tuned BART-large-CNN (406M parameters)
- ✅ **Complete MLOps Pipeline**: Data ingestion → Validation → Transformation → Training → Evaluation
- ✅ **REST API**: FastAPI with automatic documentation
- ✅ **Web Interface**: Beautiful Gradio UI for easy interaction
- ✅ **Docker Ready**: Containerized deployment with Docker Compose
- ✅ **Production Ready**: Error handling, logging, monitoring, and testing
- ✅ **Metrics Tracking**: ROUGE scores, TensorBoard integration, MLflow support

---

## 📊 Model Performance

| Metric | Score |
|--------|-------|
| **ROUGE-1** | 0.42 |
| **ROUGE-2** | 0.19 |
| **ROUGE-L** | 0.30 |
| **Compression Ratio** | ~25% |

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────────┐
│                        Text Summarizer                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Data Pipeline          Model Pipeline        Deployment       │
│  ┌──────────┐          ┌──────────┐          ┌──────────┐     │
│  │ Ingestion│──────────│ Training │──────────│ FastAPI  │     │
│  └──────────┘          └──────────┘          └──────────┘     │
│       │                     │                      │           │
│  ┌──────────┐          ┌──────────┐          ┌──────────┐     │
│  │Validation│──────────│Evaluation│──────────│  Gradio  │     │
│  └──────────┘          └──────────┘          └──────────┘     │
│       │                                           │           │
│  ┌──────────┐                                ┌──────────┐     │
│  │Transform │                                │  Docker  │     │
│  └──────────┘                                └──────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- CUDA-compatible GPU (optional, runs on CPU)
- Docker Desktop (optional, for containerization)

### Installation
```bash
# Clone repository
git clone https://github.com/yourusername/text-summarizer.git
cd text-summarizer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
pip install -e .
```

### Train Model
```bash
# Run complete training pipeline
python main.py

# This will:
# 1. Download CNN/DailyMail dataset
# 2. Validate data quality
# 3. Tokenize and preprocess
# 4. Fine-tune BART model
# 5. Evaluate performance
```

**Training Time**: ~2-4 hours on GPU, ~8-12 hours on CPU (for 1 epoch)

### Run API & Web Interface

**Option 1: Quick Start (Recommended)**
```bash
# Simple deployment script
python deploy.py

# Choose:
# 1. Docker (if Docker Desktop running)
# 2. Local (runs on localhost)
```

**Option 2: Manual Start**
```bash
# Terminal 1 - FastAPI
python app.py
# Access: http://localhost:8000

# Terminal 2 - Gradio
python gradio_app.py
# Access: http://localhost:7860
```

**Option 3: Docker Compose**
```bash
# Build and start containers
docker compose up -d

# View logs
docker compose logs -f

# Stop containers
docker compose down
```

---

## 📖 Usage Examples

### Python API
```python
from textSummarizer.components.prediction import PredictionPipeline

# Initialize pipeline
pipeline = PredictionPipeline(model_path="artifacts/model_trainer/final_model")

# Summarize text
article = """
Your long article text here...
"""

summary = pipeline.predict(
    text=article,
    max_length=128,
    min_length=30,
    num_beams=4,
    length_penalty=2.0
)

print(f"Summary: {summary}")
```

### REST API (cURL)
```bash
curl -X POST "http://localhost:8000/summarize" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Your article text here...",
    "max_length": 128,
    "min_length": 30,
    "num_beams": 4
  }'
```

### REST API (Python requests)
```python
import requests

url = "http://localhost:8000/summarize"
payload = {
    "text": "Your article text here...",
    "max_length": 128,
    "min_length": 30,
    "num_beams": 4
}

response = requests.post(url, json=payload)
result = response.json()

print(f"Summary: {result['summary']}")
print(f"Compression: {result['compression_ratio']}%")
```

---

## 📁 Project Structure
```
text-summarizer/
├── src/textSummarizer/
│   ├── components/          # Pipeline components
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   ├── model_evaluation.py
│   │   └── prediction.py
│   ├── config/              # Configuration management
│   │   └── configuration.py
│   ├── entity/              # Data classes
│   │   └── config_entity.py
│   ├── pipeline/            # Pipeline stages
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_validation.py
│   │   ├── stage_03_data_transformation.py
│   │   ├── stage_04_model_trainer.py
│   │   └── stage_05_model_evaluation.py
│   ├── utils/               # Utility functions
│   └── logging/             # Logging configuration
├── config/                  # YAML configurations
│   ├── config.yaml          # Pipeline config
│   └── params.yaml          # Training parameters
├── tests/                   # Test suite
│   ├── test_api.py
│   └── test_integration.py
├── artifacts/               # Generated artifacts
│   ├── data_ingestion/
│   ├── model_trainer/
│   └── model_evaluation/
├── app.py                   # FastAPI application
├── gradio_app.py           # Gradio interface
├── main.py                 # Main training script
├── deploy.py               # Deployment script
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Multi-container setup
├── Makefile               # Build automation
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

---

## ⚙️ Configuration

### Model Parameters (`config/params.yaml`)
```yaml
TrainingArguments:
  num_train_epochs: 3          # Training epochs
  per_device_train_batch_size: 4
  learning_rate: 5.0e-5
  warmup_steps: 500
  fp16: false                  # Use mixed precision (GPU only)
  
EvaluationArguments:
  batch_size: 8
  num_beams: 4                 # Beam search width
  max_length: 128              # Max summary length
  length_penalty: 2.0
```

### Dataset Configuration (`config/config.yaml`)
```yaml
data_ingestion:
  dataset_name: abisee/cnn_dailymail
  config_name: "3.0.0"
  split: train
  max_samples: null            # null = full dataset, or set number for testing
```

---

## 🧪 Testing
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=html

# View coverage report
open htmlcov/index.html  # Mac
start htmlcov/index.html # Windows
```

---

## 🐳 Docker Deployment

### Build Images
```bash
docker compose build
```

### Start Services
```bash
docker compose up -d
```

### Check Status
```bash
docker compose ps
docker compose logs -f
```

### Stop Services
```bash
docker compose down
```

---

## 📊 Monitoring & Metrics

### TensorBoard
```bash
# View training metrics
tensorboard --logdir artifacts/model_trainer/checkpoints
```

### MLflow (Optional)
```bash
# Start MLflow UI
mlflow ui

# Access at http://localhost:5000
```

### API Metrics

- Health check: `GET /health`
- Processing time included in API responses
- Logs stored in `logs/` directory

---

## 🔧 Development Commands

Using Make (recommended):
```bash
make help          # Show all commands
make install       # Install dependencies
make test          # Run tests
make run-api       # Start FastAPI
make run-gradio    # Start Gradio
make docker-up     # Start Docker containers
make clean         # Clean temporary files
```

---

## 📈 Performance Tips

### For Training

1. **Use GPU**: Set `fp16: true` in `params.yaml` (requires CUDA GPU)
2. **Adjust batch size**: Increase if you have more VRAM
3. **Reduce dataset**: Set `max_samples: 1000` for quick testing
4. **Use gradient accumulation**: Effective batch size = batch_size × gradient_accumulation_steps

### For Inference

1. **Use caching**: Enabled by default (`use_cache=True`)
2. **Adjust beam search**: Lower `num_beams` for faster inference
3. **Batch processing**: Process multiple texts together
4. **GPU inference**: 5-10x faster than CPU

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## 🐛 Troubleshooting

### Model Not Loading
```bash
# Check if model exists
ls artifacts/model_trainer/final_model

# If missing, train model first
python main.py
```

### Out of Memory Error
```python
# Reduce batch size in config/params.yaml
per_device_train_batch_size: 2  # Reduce from 4

# Or use gradient accumulation
gradient_accumulation_steps: 8  # Increase from 4
```

### Docker Issues
```bash
# Check Docker is running
docker --version

# Rebuild containers
docker compose down
docker compose build --no-cache
docker compose up -d
```

### Slow Training

- Use GPU: Check CUDA availability with `torch.cuda.is_available()`
- Enable FP16: Set `fp16: true` (GPU only)
- Use smaller dataset: Set `max_samples: 1000` for testing

---

## 📚 Additional Resources

- [BART Paper](https://arxiv.org/abs/1910.13461)
- [CNN/DailyMail Dataset](https://huggingface.co/datasets/abisee/cnn_dailymail)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gradio Documentation](https://gradio.app/docs/)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Md. Ashaduzzaman Sarker**
- Email: ashaduzzaman2505@gmail.com
- GitHub: [@ashaduzzaman-sarker](https://github.com/ashaduzzaman-sarker)

---

## 🙏 Acknowledgments

- [Hugging Face](https://huggingface.co/) for the Transformers library
- [Facebook AI](https://ai.facebook.com/) for the BART model
- [CNN/DailyMail](https://github.com/abisee/cnn-dailymail) dataset creators
- FastAPI and Gradio communities

---

## 📊 Project Stats

![GitHub stars](https://img.shields.io/github/stars/yourusername/text-summarizer)
![GitHub forks](https://img.shields.io/github/forks/yourusername/text-summarizer)
![GitHub issues](https://img.shields.io/github/issues/yourusername/text-summarizer)
![GitHub pull requests](https://img.shields.io/github/issues-pr/yourusername/text-summarizer)

---

## 🗺️ Roadmap

- [ ] Add multi-language support
- [ ] Implement streaming inference
- [ ] Add model quantization for edge deployment
- [ ] Create mobile app
- [ ] Add support for custom datasets
- [ ] Implement A/B testing framework
- [ ] Add Kubernetes deployment configs

---

<div align="center">

</div>
