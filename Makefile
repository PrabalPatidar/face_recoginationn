# Face Scan Project Makefile
# Provides common development and deployment commands

.PHONY: help install install-dev test lint format clean build run-web run-gui run-cli docker-build docker-run docs

# Default target
help: ## Show this help message
	@echo "Face Scan Project - Available Commands:"
	@echo "======================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# Installation commands
install: ## Install the package in production mode
	pip install -e .

install-dev: ## Install the package in development mode with all dependencies
	pip install -e ".[dev,test,docs]"
	pre-commit install

# Development commands
test: ## Run tests with coverage
	pytest --cov=src/face_scan --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without coverage (faster)
	pytest -xvs

lint: ## Run linting checks
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format: ## Format code with black and isort
	black src/ tests/
	isort src/ tests/

format-check: ## Check code formatting
	black --check src/ tests/
	isort --check-only src/ tests/

# Build and deployment
build: ## Build the package
	python -m build

build-wheel: ## Build wheel distribution
	python -m build --wheel

build-sdist: ## Build source distribution
	python -m build --sdist

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Application running
run-web: ## Run the web application
	python -m face_scan.main --mode web

run-gui: ## Run the GUI application
	python -m face_scan.main --mode gui

run-cli: ## Run the CLI application
	python -m face_scan.main --mode cli

run-dev: ## Run web app in development mode
	python -m face_scan.main --mode web --debug

# Docker commands
docker-build: ## Build Docker image
	docker build -t face-scan-project .

docker-run: ## Run Docker container
	docker run -p 5000:5000 face-scan-project

docker-dev: ## Run Docker container in development mode
	docker-compose up --build

docker-stop: ## Stop Docker containers
	docker-compose down

# Database commands
db-init: ## Initialize database
	python scripts/setup_environment.py --init-db

db-migrate: ## Run database migrations
	python scripts/setup_environment.py --migrate

db-reset: ## Reset database (WARNING: This will delete all data)
	python scripts/setup_environment.py --reset-db

# Model management
download-models: ## Download required ML models
	python scripts/download_models.py

train-model: ## Train face recognition model
	python scripts/train_model.py

# Documentation
docs: ## Generate documentation
	cd docs && make html

docs-serve: ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

# Security and quality
security-check: ## Run security checks
	bandit -r src/
	safety check

quality-check: ## Run all quality checks
	$(MAKE) lint
	$(MAKE) format-check
	$(MAKE) test
	$(MAKE) security-check

# Environment setup
setup-env: ## Set up development environment
	python scripts/setup_environment.py --full-setup

setup-data: ## Set up data directories and download sample data
	python scripts/setup_environment.py --setup-data

# Performance testing
benchmark: ## Run performance benchmarks
	python -m pytest tests/benchmark/ -v

profile: ## Profile the application
	python -m cProfile -o profile_output.prof -m face_scan.main --mode web

# Release commands
version-patch: ## Bump patch version
	bump2version patch

version-minor: ## Bump minor version
	bump2version minor

version-major: ## Bump major version
	bump2version major

release: ## Create a new release
	$(MAKE) clean
	$(MAKE) test
	$(MAKE) build
	twine upload dist/*

# Monitoring and logs
logs: ## View application logs
	tail -f logs/app.log

logs-error: ## View error logs
	tail -f logs/error.log

logs-clear: ## Clear all log files
	rm -f logs/*.log

# Backup and restore
backup: ## Create backup of data and models
	tar -czf backup_$(shell date +%Y%m%d_%H%M%S).tar.gz data/ logs/ config/

# System information
info: ## Show system and project information
	@echo "Python version:"
	@python --version
	@echo "\nPip version:"
	@pip --version
	@echo "\nInstalled packages:"
	@pip list | grep face-scan
	@echo "\nProject structure:"
	@tree -I '__pycache__|*.pyc|.git|.pytest_cache|.mypy_cache|htmlcov|dist|build|*.egg-info' -L 3

# Development workflow
dev-setup: install-dev setup-env download-models ## Complete development setup
	@echo "Development environment setup complete!"

ci: lint test security-check ## Run CI pipeline locally
	@echo "CI pipeline completed successfully!"

# Quick development commands
quick-test: ## Quick test run for development
	pytest -xvs -k "not slow"

quick-lint: ## Quick linting for development
	black --check src/ && isort --check-only src/

# Help for specific targets
help-install: ## Show installation help
	@echo "Installation Commands:"
	@echo "  make install      - Install in production mode"
	@echo "  make install-dev  - Install with development dependencies"
	@echo "  make setup-env    - Set up development environment"

help-docker: ## Show Docker help
	@echo "Docker Commands:"
	@echo "  make docker-build - Build Docker image"
	@echo "  make docker-run   - Run Docker container"
	@echo "  make docker-dev   - Run with docker-compose"

help-test: ## Show testing help
	@echo "Testing Commands:"
	@echo "  make test         - Run all tests with coverage"
	@echo "  make test-fast    - Run tests without coverage"
	@echo "  make quick-test   - Run quick tests (skip slow ones)"
