#!/usr/bin/env bash
set -euo pipefail

python3 -m pip install common/[tests]
python3 -m pip install features/
python3 -m pip install core/[all,tests]
python3 -m pip install tabular/[all,tests]
python3 -m pip install multimodal/[tests]
python3 -m pip install timeseries/[all,tests]
python3 -m pip install autogluon/

# FIXME: https://github.com/open-mmlab/mmcv/issues/3325, revert mmcv installation to use mim once fixed
python3 -m pip install "mmcv==2.1.0" --no-build-isolation --timeout 60
python3 -m pip install "mmengine==0.10.7"
python3 -m pip install --upgrade "mmdet==3.3.0"
