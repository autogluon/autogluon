#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
export AUTOGLUON_PACKAGE_NAME="autogluon-lite"

#install_local_packages "common/[tests]" "core/[all,tests]" "features/" "tabular/[all,tests]"
build_all

PYODIDE_DIR=/src/pyodide

# copy wheels to one dir to be easier loaded by pyodide.
#WHEEL_DST=$PYODIDE_DIR/wheels
WHEEL_DST=./wheels
mkdir -p $WHEEL_DST
cp -f autogluon/dist/*.whl $WHEEL_DST/
cp -f common/dist/*.whl $WHEEL_DST/
cp -f core/dist/*.whl $WHEEL_DST/
cp -f features/dist/*.whl $WHEEL_DST/
cp -f tabular/dist/*.whl $WHEEL_DST/

TEST_SCRIPT_DIR=$PYODIDE_DIR/autogluon_tests/
mkdir -p $TEST_SCRIPT_DIR
# copy test scripts to a separate dir to avoid pytest picking up test settings that conflict with pyodide.
#cp -f tabular/tests/regressiontests/*.py $TEST_SCRIPT_DIR/
#-p no:$(pwd)/tabular/tests/conftest.py \
#--dist-dir /src/pyodide \
cp -rf tabular/tests $TEST_SCRIPT_DIR/

cd $PYODIDE_DIR
#tools/pytest_wrapper.py $TEST_SCRIPT_DIR/test_tabular_lite.py -v \
#  --runtime 'firefox' --runner 'playwright' --junitxml=results.xml
tools/pytest_wrapper.py $TEST_SCRIPT_DIR/tests/regressiontests/test_tabular_lite.py -v \
  --runtime 'firefox' --runner 'playwright' --junitxml=results.xml --runpyodide
