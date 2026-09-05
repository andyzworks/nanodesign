#!/usr/bin/env bash
set -euo pipefail

python -m ruff --version
python -m ruff check --output-format=github src tests
python -m ruff format --check src tests
python -m pytest
nanodesign-v0 model-summary --config configs/v0.yaml
