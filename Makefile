# ============================================================================
# Makefile - Common Commands
# ============================================================================
.PHONY: help install test run-gradio run-api docker-build docker-up clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install all dependencies"
	@echo "  make test          - Run tests"
	@echo "  make run-gradio    - Run Gradio interface"
	@echo "  make run-api       - Run FastAPI server"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start Docker services"
	@echo "  make clean         - Clean temporary files"

install:
	pip install -r requirements.txt
	pip install -r requirements-api.txt

test:
	pytest tests/ -v

run-gradio:
	python app.py

run-api:
	uvicorn src.textSummarizer.api.app:app --reload --host 0.0.0.0 --port 8000

docker-build:
	docker-compose build

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-logs:
	docker-compose logs -f

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage

format:
	black src/ tests/
	isort src/ tests/

lint:
	flake8 src/ tests/
	mypy src/
