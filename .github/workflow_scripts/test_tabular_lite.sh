#!/bin/bash
# TODO: re-enable when pandas 2.0 supported in browser

# set -ex

# source $(dirname "$0")/env_setup.sh

# setup_build_env
# export AUTOGLUON_PACKAGE_NAME="autogluon-lite"

# install_local_packages "common/[tests]" "core/[all,tests]" "features/" "tabular/[all,tests]"
# build_pkg "common" "core" "features" "tabular" "autogluon"

# PYODIDE_DIR=/src/pyodide

# cd $PYODIDE_DIR
# tools/pytest_wrapper.py autogluon/tabular/tests/regressiontests/test_tabular_lite.py -v \
#   --runtime 'firefox' --runner 'playwright' --junitxml=results.xml --runpyodide
