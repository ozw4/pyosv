#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

python -m pytest -q
python -m ruff check src tests examples
python -m ruff format --check src tests examples
