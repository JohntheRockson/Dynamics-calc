# Lightweight local stand-ins for the GitHub Actions jobs in .github/workflows/ci.yml
PYTHON ?= python3

.PHONY: test lint typecheck coverage build ci

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check src tests

typecheck:
	$(PYTHON) -m mypy src

coverage:
	$(PYTHON) -m pytest --cov=attitude_sim --cov-report=term-missing --cov-report=xml

build:
	$(PYTHON) -m build

ci: lint typecheck test
