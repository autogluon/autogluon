#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh
source $(dirname "$0")/build_tabular_lite.sh

setup_build_env
build_tabular_lite

PYODIDE_DIR=/src/pyodide

cd $PYODIDE_DIR
tools/pytest_wrapper.py autogluon/tabular/tests/regressiontests/test_tabular_lite.py -v \
  --runtime 'firefox' --runner 'playwright' --junitxml=results.xml --runpyodide
