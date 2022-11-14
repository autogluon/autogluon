#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
python3 -m pip install 'mxnet==1.9.*'
install_all
build_all
