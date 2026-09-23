.PHONY: install lint typecheck test security check demo studio dist release-check release-audit docker-up docker-down

PYTHON ?= $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)

install:
	$(PYTHON) -m pip install -e ".[dev,studio,research,signing]"

lint:
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check src tests examples

typecheck:
	$(PYTHON) -m mypy src examples/offline_pipeline.py

test:
	$(PYTHON) -m pytest --cov=agentic_rl_forge --cov-report=term-missing

security:
	$(PYTHON) -m pip check
	$(PYTHON) -m pip_audit

check: lint typecheck test
	bash -n recipes/verl/run_search_r1_grpo.sh

demo:
	$(PYTHON) -m agentic_rl_forge.cli demo

studio:
	$(PYTHON) -m agentic_rl_forge.cli studio

dist:
	$(PYTHON) -m build
	$(PYTHON) -m twine check --strict dist/*

release-check: check security dist
	$(PYTHON) -m agentic_rl_forge.cli doctor --profile core --project . --strict
	$(PYTHON) -m agentic_rl_forge.cli demo

release-audit:
	$(PYTHON) -m agentic_rl_forge.cli release-audit --project . --run-checks --build-wheel

docker-up:
	docker compose up --build

docker-down:
	docker compose down
