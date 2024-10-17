#!/usr/bin/env bash
set -euo pipefail

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Installing uv..."
    python3 -m pip install uv
fi

# Use uv to install packages
uv pip install -e common/[tests] --no-cache-dir
uv pip install -e core/[all,tests] --no-cache-dir
uv pip install -e features/ --no-cache-dir
uv pip install -e tabular/[all,tests] --no-cache-dir
uv pip install -e multimodal/[tests] --no-cache-dir
uv pip install -e timeseries/[all,tests] --no-cache-dir
uv pip install -e eda/ --no-cache-dir
uv pip install -e autogluon/ --no-cache-dir
