#!/usr/bin/env bash
set -euo pipefail
python3 -m pip install -e core/
python3 -m pip install -e features/
python3 -m pip install -e tabular/[all]
python3 -m pip install -e mxnet/
python3 -m pip install -e extra/
python3 -m pip install -e text/
python3 -m pip install -e vision/
python3 -m pip install -e forecasting/
python3 -m pip install -e tabular_to_image/
python3 -m pip install -e autogluon/
python3 -m pip install  pydot
python3 -m pip install pydotplus
sudo apt-get install graphviz libgraphviz-dev pkg-config
sudo apt-get install python-pip python-virtualenv
python3 -m pip install  pygraphviz
python3 -m pip install -e DeepInsight/