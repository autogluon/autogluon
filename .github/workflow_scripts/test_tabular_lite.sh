#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
export AUTOGLUON_PACKAGE_NAME="autogluon-lite"

build_all

PYODIDE_DIR=/src/pyodide

DIST_DST=$PYODIDE_DIR/wheels
mkdir -p $DIST_DST
cp -f autogluon/dist/*.whl $DIST_DST/
cp -f common/dist/*.whl $DIST_DST/
cp -f core/dist/*.whl $DIST_DST/
cp -f features/dist/*.whl $DIST_DST/
cp -f tabular/dist/*.whl $DIST_DST/

#cp -f tabular/tests/regressiontests/test_tabular_lite*.py $PYODIDE_DIR/

cd $PYODIDE_DIR
tools/pytest_wrapper.py /src/autogluon/tabular/tests/regressiontests/test_tabular_lite2.py -v \
  --runtime 'firefox' --runner 'playwright' --junitxml=results.xml
