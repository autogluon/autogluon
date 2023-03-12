#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
export AUTOGLUON_PACKAGE_NAME="autogluon-lite"

#install_local_packages "common/[tests]" "core/[all,tests]" "features/" "tabular/[all,tests]"
build_all

PYODIDE_DIR=/src/pyodide

DIST_DST=$PYODIDE_DIR/wheels
mkdir -p $DIST_DST
cp -f autogluon/dist/*.whl $DIST_DST/
cp -f common/dist/*.whl $DIST_DST/
cp -f core/dist/*.whl $DIST_DST/
cp -f features/dist/*.whl $DIST_DST/
cp -f tabular/dist/*.whl $DIST_DST/

TEST_SCRIPT_DIR=$PYODIDE_DIR/autogluon_tests/
mkdir -p $TEST_SCRIPT_DIR
cp -f tabular/tests/regressiontests/*.py $TEST_SCRIPT_DIR/

cd $PYODIDE_DIR
tools/pytest_wrapper.py $TEST_SCRIPT_DIR/test_tabular_lite2.py -v \
  --runtime 'firefox' --runner 'playwright' --junitxml=results.xml
