PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
export PYTHONPATH := src

.PHONY: setup test reproduce data-check download clean notebooks

setup:
	@if [ "$$(uname)" = Darwin ] && command -v brew >/dev/null 2>&1; then \
		brew list libomp >/dev/null 2>&1 || brew install libomp; \
	fi
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.13 $(VENV); \
		uv pip install --python $(BIN)/python -e ".[dev,notebook,full]"; \
	else \
		$(PYTHON) -m venv $(VENV); \
		$(BIN)/pip install -U pip; \
		$(BIN)/pip install -e ".[dev,notebook,full]"; \
	fi

test:
	$(BIN)/pytest

download:
	$(BIN)/python scripts/download_data.py

data-check:
	$(BIN)/python -m macro_stress.cli data-check

reproduce:
	$(BIN)/python -m macro_stress.cli reproduce

notebooks:
	$(BIN)/python -m jupyter execute notebooks/00_index.ipynb notebooks/01_milestone1_baselines.ipynb notebooks/02_milestone2_safehaven_and_modeling.ipynb notebooks/03_milestone3_interpretability_and_robustness.ipynb notebooks/04_milestone3_direct_safehaven_findings.ipynb --inplace

clean:
	rm -rf $(VENV) .pytest_cache src/*.egg-info *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
