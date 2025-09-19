# Qugeister Development Makefile

.PHONY: help install install-dev clean test lint format type-check docs build upload

# Default target
help:
	@echo "Available commands:"
	@echo "  install      Install package in production mode"
	@echo "  install-dev  Install package in development mode"
	@echo "  clean        Clean build artifacts and cache"
	@echo "  test         Run test suite"
	@echo "  lint         Run code linting"
	@echo "  format       Format code with black"
	@echo "  type-check   Run mypy type checking"  
	@echo "  docs         Build documentation"
	@echo "  build        Build distribution packages"
	@echo "  upload       Upload to PyPI"

# Installation
install:
	pip install .

install-dev:
	pip install -e ".[dev,docs]"
	pre-commit install

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Development
test:
	pytest tests/ --cov=src/qugeister --cov-report=html --cov-report=term-missing

lint:
	flake8 src/ tests/ scripts/

format:
	black src/ tests/ scripts/
	
type-check:
	mypy src/qugeister

# Documentation
docs:
	cd docs && make html

# Distribution  
build: clean
	python -m build

upload: build
	python -m twine upload dist/*

# Training shortcuts
train-quick:
	python scripts/train.py --episodes 100 --qubits 4

train-full:
	python scripts/train.py --episodes 10000 --qubits 4

# Analysis shortcuts  
analyze-quick:
	python scripts/analyze.py --states 100

analyze-full:
	python scripts/analyze.py --states 5000

# Web interface
web-designer:
	qugeister web --mode designer

web-playground:
	qugeister web --mode playground

# Tournament
tournament:
	qugeister tournament --rounds 10

# Development workflow
dev-setup: install-dev
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make train-quick' for quick training"

dev-check: lint type-check test
	@echo "All checks passed!"

# CI/CD targets
ci-test:
	pytest tests/ --cov=src/qugeister --cov-report=xml --cov-fail-under=80

ci-build: clean build
	@echo "Build completed successfully"