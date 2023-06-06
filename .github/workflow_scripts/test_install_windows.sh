#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
setup_torch_cpu
install_all_windows
build_all
