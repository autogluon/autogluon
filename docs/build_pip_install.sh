#!/bin/bash
python3 -m pip uninstall -y autogluon
python3 -m pip uninstall -y autogluon.vision
python3 -m pip uninstall -y autogluon.text
python3 -m pip uninstall -y autogluon.tabular
python3 -m pip uninstall -y autogluon.timeseries
python3 -m pip uninstall -y autogluon.features
python3 -m pip uninstall -y autogluon.core
python3 -m pip uninstall -y autogluon.common
python3 -m pip uninstall -y autogluon-contrib-nlp

cd common/
python3 -m pip install -e .
cd ..

cd core/
python3 -m pip install -e .[all]
cd ..

cd features/
python3 -m pip install -e .
cd ..

cd tabular/
# Python 3.7 bug workaround: https://github.com/python/typing/issues/573
python3 -m pip uninstall -y typing
python3 -m pip install -e .[all,tests]
cd ..

cd text/
python3 -m pip install -e .
cd ..

cd vision/
python3 -m pip install -e .
cd ..

cd timeseries/
python3 -m pip install -e .
cd ..

cd autogluon/
python3 -m pip install -e .
cd ..
