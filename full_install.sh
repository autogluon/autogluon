#!/usr/bin/env bash
set -euo pipefail

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Installing uv..."
    python3 -m pip install uv
fi

# Use uv to install packages
python3 -m uv pip install --refresh -e common/[tests] -e core/[all,tests] -e features/ -e tabular/[all,tests] -e multimodal/[tests] -e timeseries/[all,tests] -e eda/ -e autogluon/
