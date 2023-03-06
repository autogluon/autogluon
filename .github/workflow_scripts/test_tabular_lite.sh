#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
export AUTOGLUON_PACKAGE_NAME="autogluon-lite"

build_all

DIST_DST=/pyodide/wheels
mkdir -p $DIST_DST
cp -f autogluon/dist/*.whl $DIST_DST/
cp -f common/dist/*.whl $DIST_DST/
cp -f core/dist/*.whl $DIST_DST/
cp -f features/dist/*.whl $DIST_DST/
cp -f tabular/dist/*.whl $DIST_DST/

cp -f tabular/tests/regressiontests/test_tabular_lite.py /pyodide/

cd /pyodide
pwd
ls
#tools/pytest_wrapper.py test_tabular_lite -v \
#  --runtime 'firefox' --runner 'playwright' --junitxml=results.xml
