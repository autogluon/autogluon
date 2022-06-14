#!/bin/bash

set -ex

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4  # For push events, PR_NUMBER will be empty

source $(dirname "$0")/env_setup.sh
source $(dirname "$0")/write_to_s3.sh

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

if [[ -n $PR_NUMBER ]]; then BUCKET=autogluon-ci S3_PATH=s3://$BUCKET/build_docs/$PR_NUMBER/$COMMIT_SHA; else BUCKET=autogluon-ci-push S3_PATH=s3://$BUCKET/build_docs/$BRANCH/$COMMIT_SHA; fi
DOC_PATH=docs/_build/rst/tutorials/cloud_fit_deploy/
S3_PATH=$BUCKET/cloud_fit_deploy/

write_to_s3 $BUCKET $DOC_PATH $S3_PATH
