#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: Python 3.10 is required."; exit 1; }
[ -x .venv/bin/python ] || "$PYTHON_BIN" -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-linux-cpu.txt
python verify_environment.py
exec python app.py
