#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
setup_torch_cpu
python3 -m pip install 'mxnet==1.9.*'
install_all
build_all
