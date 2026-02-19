.PHONY: setup test smoke data run analyze clean

# --- Setup ---
setup:
	python -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt
	.venv/bin/python -m spacy download en_core_web_sm
	@echo "✓ Setup complete. Run: source .venv/bin/activate"

# --- Test ---
test:
	python tests/test_smoke.py

# --- Pipeline ---
data:
	python -m src.prepare_data

smoke:
	python -m src.run_judge --smoke

run:
	python -m src.run_judge

analyze:
	python -m src.analyze

# --- Clean ---
clean:
	rm -rf __pycache__ src/__pycache__ tests/__pycache__
	rm -rf .pytest_cache
