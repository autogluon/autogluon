#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env
bandit -r multimodal/src -ll
black --check --diff multimodal/
isort --check --diff multimodal/
black --check --diff timeseries/
isort --check --diff timeseries/
