#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -e core/
python3 -m pip install -e features/
python3 -m pip install -e tabular/[all]
python3 -m pip install -e forecasting/[all]
python3 -m pip install -e mxnet/
python3 -m pip install -e extra/
python3 -m pip install -e text/
python3 -m pip install -e vision/
python3 -m pip install -e autogluon/
