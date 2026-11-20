.PHONY: dev test

.venv/bin/python:
	python3 -m venv .venv
	.venv/bin/python -m pip install -r requirements.txt -r requirements-dev.txt

dev: .venv/bin/python
	.venv/bin/python -c "import nltk; nltk.download('punkt_tab', quiet=True); nltk.download('averaged_perceptron_tagger_eng', quiet=True)"
	.venv/bin/python scripts/ensure_dev_env.py
	EMAIL_PROVIDER=dry-run .venv/bin/python -m uvicorn app.main:create_app --factory --reload --host 127.0.0.1 --port 8000

test: .venv/bin/python
	.venv/bin/python -m pytest
