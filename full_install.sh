# ConfigSpace must be built against correct numpy and scipy; by default build isolation will force versions
# specified in https://github.com/automl/ConfigSpace/blob/master/pyproject.toml
# This fix does two things:
# 1) it forces ConfigSpace to ignore toml dependencies (numpy 1.20.x)
# 2) it forces re-install of ConfigSpace in case it was cached before
python3 -m pip uninstall -y ConfigSpace
python3 -m pip install 'numpy==1.19.5'
python3 -m pip install 'Cython>=0.29.21,<3'
python3 -m pip install --force 'ConfigSpace==0.4.14' --no-binary :all:

# Normal installation
python3 -m pip install -e core/
python3 -m pip install -e features/
python3 -m pip install -e tabular/
python3 -m pip install -e mxnet/
python3 -m pip install -e extra/
python3 -m pip install -e text/
python3 -m pip install -e vision/
python3 -m pip install -e autogluon/
