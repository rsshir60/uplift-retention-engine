.PHONY: install train evaluate serve test lint clean

# ── Environment ────────────────────────────────────────────────────────────
install:
	pip install -r requirements.txt

# ── Pipeline ───────────────────────────────────────────────────────────────
train:
	python scripts/train.py --source synthetic --learners x_learner dr_learner

train-criteo:
	python scripts/train.py --source criteo --sample-frac 0.05

evaluate:
	python scripts/evaluate.py --model x_learner --source synthetic

# ── Serving ────────────────────────────────────────────────────────────────
serve:
	uvicorn src.api.main:app --reload --port 8000

serve-docker:
	docker compose up --build

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# ── Testing ────────────────────────────────────────────────────────────────
test:
	pytest tests/ -v --asyncio-mode=auto

test-cov:
	pytest tests/ -v --asyncio-mode=auto --cov=src --cov-report=term-missing

# ── Code quality ───────────────────────────────────────────────────────────
lint:
	ruff check src/ scripts/ tests/
	mypy src/ --ignore-missing-imports

# ── Cleanup ────────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache .mypy_cache .ruff_cache
