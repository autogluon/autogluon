#!/usr/bin/env bash
set -euo pipefail
uv pip install -e common/[tests]
uv pip install -e core/[all,tests]
uv pip install -e features/
uv pip install -e tabular/[all,tests]
uv pip install -e multimodal/[tests]
uv pip install -e timeseries/[all,tests]
uv pip install -e eda/
uv pip install -e autogluon/
