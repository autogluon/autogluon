#!/usr/bin/env bash
set -euo pipefail

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Installing uv..."
    python3 -m pip install uv
fi

# Use uv to install packages
uv pip install -e common/[tests]
uv pip install -e core/[all,tests]
uv pip install -e features/
uv pip install -e tabular/[all,tests]
uv pip install -e multimodal/[tests]
uv pip install -e timeseries/[all,tests]
uv pip install -e eda/
uv pip install -e autogluon/
