#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -c 'import sys; assert sys.version_info[:2]==(3,10), "Python 3.10 is required"'
./repair_dgx_opencv.sh
python3 -m pip install -r requirements-dgx.txt
python3 verify_environment.py --dgx
exec python3 app.py
