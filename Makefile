.PHONY: help install test docker clean

help:
	@echo "Available commands:"
	@echo "  make install       - Install all dependencies"
	@echo "  make main       	- Run all pipelines"
	@echo "  make test          - Run tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make docker-build  - Build Docker images"
	@echo "  make docker-up     - Start all services"
	@echo "  make docker-down   - Stop all services"
	@echo "  make docker-logs   - Show container logs"
	@echo "  make run-api       - Run API locally"
	@echo "  make run-ui        - Run Gradio UI locally"
	@echo "  make clean         - Clean temporary files"

install: 
	pip install -r requirements.txt
	pip install -r requirements-test.txt
	pip install -e .

main:
	python main.py

test: 
	pytest tests/ -v

test-cov:  
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term
	@echo "\nCoverage report: htmlcov/index.html"

docker-build: 
	docker-compose build

docker-up: 
	docker-compose up -d
	@echo "\nAPI: http://localhost:8000"
	@echo "UI:  http://localhost:7860"

docker-down: 
	docker-compose down

docker-logs: 
	docker-compose logs -f

run-api:  
	python app.py

run-ui:  
	python gradio_app.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .coverage build dist
	@echo "Cleaned temporary files"