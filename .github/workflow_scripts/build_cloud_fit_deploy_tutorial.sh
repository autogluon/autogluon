#!/bin/bash

set -ex

PR_NUMBER=$(basename $1) # For push events, this will be master branch instead of PR number
COMMIT_SHA=$2

source $(dirname "$0")/env_setup.sh

setup_build_contrib_env
bash docs/build_pip_install.sh
# only build for docs/cloud_fit_deploy
shopt -s extglob
rm -rf ./docs/tutorials/!(cloud_fit_deploy)
cd docs && rm -rf _build && d2lbook build rst

COMMAND_EXIT_CODE=$?
if [ $COMMAND_EXIT_CODE -ne 0 ]; then
    exit COMMAND_EXIT_CODE
fi

cd ..
aws s3 cp --recursive docs/_build/rst/tutorials/cloud_fit_deploy/ s3://autogluon-dev/build_docs/$PR_NUMBER/$COMMIT_SHA/cloud_fit_deploy/ --quiet
