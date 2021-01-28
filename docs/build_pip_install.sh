#!/bin/bash
python3 -m pip uninstall -y scipy
python3 -m pip install git+https://github.com/zhanghang1989/d2l-book
python3 -m pip install --force-reinstall ipython==7.16

python3 -m pip uninstall -y autogluon
python3 -m pip uninstall -y autogluon.vision
python3 -m pip uninstall -y autogluon.text
python3 -m pip uninstall -y autogluon.mxnet
python3 -m pip uninstall -y autogluon.extra
python3 -m pip uninstall -y autogluon.tabular
python3 -m pip uninstall -y autogluon.features
python3 -m pip uninstall -y autogluon.core
python3 -m pip uninstall -y autogluon-contrib-nlp

cd core/
python3 -m pip install --upgrade -e .
cd ..

cd features/
python3 -m pip install --upgrade -e .
cd ..

cd tabular/
# Python 3.7 bug workaround: https://github.com/python/typing/issues/573
python3 -m pip uninstall -y typing
python3 -m pip install --upgrade -e .
cd ..

cd mxnet/
python3 -m pip install --upgrade -e .
cd ..

cd extra/
python3 -m pip install --upgrade -e .
cd ..

cd text/
python3 -m pip install --upgrade -e .
cd ..

cd vision/
python3 -m pip install --upgrade -e .
cd ..

cd autogluon/
python3 -m pip install --upgrade -e .
cd ..
