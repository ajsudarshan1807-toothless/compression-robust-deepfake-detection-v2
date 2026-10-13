.PHONY: setup lint test data train eval app clean

PYTHON ?= python

setup:
	$(PYTHON) -m pip install -e ".[dev]"

lint:
	ruff check deepfake_detector tests
	ruff format --check deepfake_detector tests

format:
	ruff format deepfake_detector tests
	ruff check --fix deepfake_detector tests

test:
	pytest -v tests

data:
	$(PYTHON) -m deepfake_detector.data.datasets

train:
	$(PYTHON) -m deepfake_detector.train.train_image

eval:
	$(PYTHON) -m deepfake_detector.eval.evaluate

app:
	uvicorn apps.api.main:app --reload --port 8000

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache
