#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
install_common
install_features

cd features/
python3 -m pytest --junitxml=results.xml --runslow tests
