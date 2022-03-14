#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
install_core_all_tests

cd core/
python3 -m pytest --junitxml=results.xml --runslow tests
