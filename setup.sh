#!/bin/bash
# =============================================================
# LLM-Judge Semantic Uncertainty - 원클릭 환경 세팅
# Usage: bash setup.sh
# =============================================================
set -e

echo "=== LLM-Judge Semantic Uncertainty Setup ==="

# 1. Python venv
echo "[1/5] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
echo "[2/5] Installing dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# 3. spaCy model
echo "[3/5] Downloading spaCy English model..."
python -m spacy download en_core_web_sm -q

# 4. .env
echo "[4/5] Checking .env..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  → .env created from template. Please add your OPENAI_API_KEY."
else
    echo "  → .env already exists."
fi

# 5. Smoke test
echo "[5/5] Running smoke test..."
python tests/test_smoke.py

echo ""
echo "=== Setup Complete ==="
echo "Next steps:"
echo "  1. Edit .env with your OPENAI_API_KEY"
echo "  2. source .venv/bin/activate"
echo "  3. python -m src.prepare_data"
