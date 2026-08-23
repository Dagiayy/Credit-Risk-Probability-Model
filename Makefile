.PHONY: install test lint format train predict api mlflow docker-up docker-down

install:
	pip install -r requirements-dev.txt

test:
	pytest tests/ --cov=src --cov-report=term-missing

lint:
	flake8 src/ tests/

format:
	black src/ tests/

train:
	python -m src.train

predict:
	python -m src.predict --input data/raw/data.csv --output reports/risk_scores.csv

mlflow:
	mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

api:
	uvicorn src.api.main:app --reload

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down
