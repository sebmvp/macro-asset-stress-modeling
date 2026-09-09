PYTHON ?= python3
VENV := .venv
BIN := $(VENV)/bin
export PYTHONPATH := src

.PHONY: setup test reproduce data-check clean

setup:
	@if [ "$$(uname)" = Darwin ] && command -v brew >/dev/null 2>&1; then \
		brew list libomp >/dev/null 2>&1 || brew install libomp; \
	fi
	@if command -v uv >/dev/null 2>&1; then \
		uv venv --python 3.13 $(VENV); \
		uv pip install --python $(BIN)/python -e ".[dev]"; \
	else \
		$(PYTHON) -m venv $(VENV); \
		$(BIN)/pip install -U pip; \
		$(BIN)/pip install -e ".[dev]"; \
	fi

test:
	$(BIN)/pytest

data-check:
	$(BIN)/python -m macro_stress.cli data-check

reproduce:
	$(BIN)/python -m macro_stress.cli reproduce

clean:
	rm -rf $(VENV) .pytest_cache src/*.egg-info *.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
