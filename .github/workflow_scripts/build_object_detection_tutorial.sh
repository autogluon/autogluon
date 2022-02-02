#!/bin/bash

set -ex

PR_NUMBER=$(basename $1) # For push events, this will be master branch instead of PR number
COMMIT_SHA=$2

source $(dirname "$0")/env_setup.sh

setup_build_contrib_env
setup_mxnet_gpu
export CUDA_VISIBLE_DEVICES=0
bash docs/build_pip_install.sh
# only build for docs/object_detection
shopt -s extglob
rm -rf ./docs/tutorials/!(object_detection)
cd docs && rm -rf _build && d2lbook build rst && cd ..

aws s3 cp --recursive docs/_build/rst/tutorials/object_detection/ s3://autogluon-dev/build_docs/$PR_NUMBER/$COMMIT_SHA/ --quiet
