#!/bin/bash

set -ex

source $(dirname "$0")/env_setup.sh

setup_build_env

cd cloud/
python3 -m tox -e format
python3 -m tox -e lint
python3 -m tox -e isort
