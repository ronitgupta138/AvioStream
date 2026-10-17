.PHONY: all build test test-py test-java run-dashboard run-pipeline clean docker-build docker-run

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
STREAMLIT := .venv/bin/streamlit
MVN := mvn

all: test

build:
	@echo "===> Compiling Java 21 Project Loom Streaming Engine..."
	cd src/java_engine && $(MVN) clean compile -B

test: test-py test-java
	@echo "===> All test suites verified successfully!"

test-py:
	@echo "===> Running Pytest Suite (23 Tests)..."
	PYTHONPATH=. $(PYTEST) tests/ -v

test-java:
	@echo "===> Running Java 21 Engine Test Suite..."
	cd src/java_engine && $(MVN) test -B

run-dashboard:
	@echo "===> Launching StrataSense Streamlit Operations Console..."
	$(STREAMLIT) run src/dashboard/app.py

run-pipeline:
	@echo "===> Executing Complete End-to-End Prognostics Pipeline..."
	PYTHONPATH=. $(PYTHON) src/run_pipeline.py --all

docker-build:
	@echo "===> Building Production Multi-Stage Container..."
	docker build -t stratasense:latest .

docker-run:
	@echo "===> Running StrataSense Container on port 8501..."
	docker run -d -p 8501:8501 --name stratasense-fleet stratasense:latest

clean:
	@echo "===> Cleaning target directories and Python bytecode..."
	cd src/java_engine && $(MVN) clean
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache
