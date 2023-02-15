#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install common/[tests]
python3 -m pip install core/[all,tests]
python3 -m pip install features/
python3 -m pip install tabular/[all,tests]
python3 -m pip install multimodal/[tests]
python3 -m pip install timeseries/[all,tests]
python3 -m pip install autogluon/
