#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install common/[tests]
python3 -m pip install core/[all,tests]
python3 -m pip install features/
python3 -m pip install tabular/[all,tests]
python3 -m pip install multimodal/[tests]
python3 -m pip install timeseries/[all,tests]
python3 -m pip install autogluon/

mim install mmcv --timeout 60
python3 -m pip install --upgrade "mmdet>=3.0.0"
# python3 -m pip install --upgrade "mmocr<1.0"  # not compatible with mmcv 2.0
