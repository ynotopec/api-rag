PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
UVICORN := $(VENV)/bin/uvicorn

.PHONY: setup run test clean

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -U pip
	$(PIP) install -r requirements.txt

run:
	$(UVICORN) app:app --host 0.0.0.0 --port $${PORT:-8080}

test:
	$(PY) -m compileall app.py src evaluate_rag.py
	$(PY) evaluate_rag.py --help >/dev/null

clean:
	rm -rf $(VENV) __pycache__ src/**/__pycache__
