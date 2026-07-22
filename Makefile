.PHONY: setup generate-data validate-data prepare-data train evaluate build-index serve test lint format typecheck security docs demo smoke-test

CONFIG ?= configs/demo.yaml

setup:
	uv sync --all-extras

generate-data:
	uv run recommender generate-data --config $(CONFIG)

validate-data:
	uv run recommender validate-data --config $(CONFIG)

prepare-data:
	uv run recommender prepare-data --config $(CONFIG)

train:
	uv run recommender train --config $(CONFIG)

evaluate:
	uv run recommender evaluate --config $(CONFIG)

build-index:
	uv run recommender build-index --config $(CONFIG)

serve:
	uv run recommender serve --config $(CONFIG)

test:
	uv run pytest --cov --cov-branch --cov-report=term-missing

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run mypy

security:
	uv run bandit -c pyproject.toml -r src
	uv run detect-secrets scan --baseline .secrets.baseline
	uv run pip-audit

docs:
	uv run mkdocs build --strict

demo:
	uv run recommender run-pipeline --config configs/demo.yaml

smoke-test:
	uv run recommender smoke-test --config $(CONFIG)
