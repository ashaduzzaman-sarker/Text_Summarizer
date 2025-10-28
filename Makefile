# ============================================================================
# Makefile - Quick Commands
# ============================================================================

.PHONY: help install test docker clean

help:  ## Show this help message
	@echo "Available commands:"
	@echo "  make install       - Install all dependencies"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start all services"
	@echo "  make docker-down   - Stop all services"
	@echo "  make docker-logs   - Show container logs"
	@echo "  make run-api       - Run API locally"
	@echo "  make run-ui        - Run Gradio UI locally"
	@echo "  make clean         - Clean temporary files"

install:  ## Install dependencies
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	pip install -e .

test:  ## Run tests
	pytest tests/ -v

test-cov:  ## Run tests with coverage
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "\nCoverage report: htmlcov/index.html"

docker-build:  ## Build Docker images
	docker-compose build

docker-up:  ## Start Docker containers
	docker-compose up -d
	@echo "\nAPI: http://localhost:8000"
	@echo "UI:  http://localhost:7860"

docker-down:  ## Stop Docker containers
	docker-compose down

docker-logs:  ## Show container logs
	docker-compose logs -f

run-api:  ## Run API locally
	python app.py

run-ui:  ## Run Gradio UI locally
	python gradio_app.py

clean:  ## Clean temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage build dist
	@echo "Cleaned temporary files"