#!/usr/bin/env bash
set -euo pipefail

# Check if uv is installed
if ! command -v uv &> /dev/null
then
    echo "uv could not be found. Installing uv..."
    python3 -m pip install uv
fi

# Use uv to install packages
uv pip install -e common/[tests] core/[all,tests] features/ tabular/[all,tests] multimodal/[tests] timeseries/[all,tests] eda/ autogluon/
