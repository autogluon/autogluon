#!/bin/bash
python3 -m pip uninstall -y autogluon
python3 -m pip uninstall -y autogluon.fair
python3 -m pip uninstall -y autogluon.eda
python3 -m pip uninstall -y autogluon.timeseries
python3 -m pip uninstall -y autogluon.multimodal
python3 -m pip uninstall -y autogluon.tabular
python3 -m pip uninstall -y autogluon.core
python3 -m pip uninstall -y autogluon.features
python3 -m pip uninstall -y autogluon.common

cd common/
python3 -m pip install -e .
cd ..

cd features/
python3 -m pip install -e .
cd ..

cd core/
python3 -m pip install -e .[all]
cd ..

cd tabular/
# Python 3.7 bug workaround: https://github.com/python/typing/issues/573
python3 -m pip uninstall -y typing
python3 -m pip install -e .[all,tests]
cd ..

cd multimodal/
python3 -m pip install -e .
cd ..

cd timeseries/
python3 -m pip install -e .[all,tests]
cd ..

cd eda/
python3 -m pip install -e .[tests]
# Resolve awscli and tox conflict
python3 -m pip install "colorama<0.4.5,>=0.2.5"
cd ..

cd autogluon/
python3 -m pip install -e .
cd ..
