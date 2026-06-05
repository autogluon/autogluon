#!/bin/bash
python3 -m pip uninstall -y autogluon
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
python3 -m pip install -e .[all,tests]
cd ..

cd multimodal/
python3 -m pip install -e .
cd ..

cd timeseries/
python3 -m pip install -e .[all,tests]
cd ..

cd autogluon/
python3 -m pip install -e .
cd ..
