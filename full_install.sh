#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -e common/[tests]
python3 -m pip install -e core/[all,tests]
python3 -m pip install -e features/
python3 -m pip install -e tabular/[all,tests]
python3 -m pip install -e multimodal/[tests]
python3 -m pip install -e text/[tests]
python3 -m pip install -e vision/
python3 -m pip install -e timeseries/[all,tests]
python3 -m pip install -e eda/
python3 -m pip install -e fair/
python3 -m pip install -e autogluon/
