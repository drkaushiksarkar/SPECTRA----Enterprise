.PHONY: setup test lint format build

setup:
	python -m venv .venv && . .venv/bin/activate && pip install -U pip && pip install -e .

lint:
	. .venv/bin/activate && python -m pip install black isort ruff && ruff check spectra_dx && black --check spectra_dx && isort --check-only spectra_dx

format:
	. .venv/bin/activate && python -m pip install black isort && black spectra_dx && isort spectra_dx

test:
	. .venv/bin/activate && python -m pip install pytest && pytest -q

build:
	python -m build
