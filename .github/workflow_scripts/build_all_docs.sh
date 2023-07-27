#!/usr/bin/env bash

# This script build the docs and store the results into a intermediate bucket to prevent our web hosting bucket being manipulated intentionally
# The final docs will be copied to the web hosting bucket diromg GitHub workflow that runs in the context of the base repository's default branch

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

set -ex

source $(dirname "$0")/env_setup.sh
source $(dirname "$0")/write_to_s3.sh

if [[ (-n $PR_NUMBER) || ($GIT_REPO != autogluon/autogluon) ]]
then
    bucket='autogluon-staging'
    if [[ -n $PR_NUMBER ]]; then path=$PR_NUMBER; else path=$BRANCH; fi
    site=$bucket.s3-website-us-west-2.amazonaws.com/$path/$COMMIT_SHA  # site is the actual bucket location that will serve the doc
else
    if [[ $BRANCH == 'master' ]]
    then
        path='dev'
    else
        if [[ $BRANCH == 'dev' ]]
        then
            path='dev-branch'
        else
            path=$BRANCH
        fi
    fi
    bucket='autogluon.mxnet.io'
    site=$bucket/$path  # site is the actual bucket location that will serve the doc
fi

other_doc_version_text='Stable Version Documentation'
other_doc_version_branch='stable'
if [[ $BRANCH == 'stable' ]]
then
    other_doc_version_text='Dev Version Documentation'
    other_doc_version_branch='dev'
fi

if [[ -n $PR_NUMBER ]]; 
then 
    BUCKET=autogluon-ci
    BUILD_DOCS_PATH=s3://$BUCKET/build_docs/$PR_NUMBER/$COMMIT_SHA
    S3_PATH=s3://$BUCKET/build_docs/${path}/$COMMIT_SHA/all
else
    BUCKET=autogluon-ci-push
    BUILD_DOCS_PATH=s3://$BUCKET/build_docs/$BRANCH/$COMMIT_SHA
    S3_PATH=s3://$BUCKET/build_docs/$BRANCH/$COMMIT_SHA/all  # We still write to BRANCH so copy_docs.sh knows where to find it
fi

setup_build_contrib_env
install_all_no_tests

LOCAL_DOC_PATH=_build/html

cd docs
rm -rf tutorials/eda  # disable eda temporarily
sphinx-build -D nb_execution_mode=off -b html . $LOCAL_DOC_PATH

rm -rf "$LOCAL_DOC_PATH/.doctrees/" # remove build artifacts that are not needed to serve webpage

# Overwrite un-executed tutorials w/ executed versions (with images) from other build jobs
aws s3 cp $BUILD_DOCS_PATH/tutorials/ $LOCAL_DOC_PATH/tutorials/ --recursive --exclude "*/index.html"
aws s3 cp $BUILD_DOCS_PATH/_images/ $LOCAL_DOC_PATH/_images/ --recursive

COMMAND_EXIT_CODE=$?
if [ $COMMAND_EXIT_CODE -ne 0 ]; then
    exit COMMAND_EXIT_CODE
fi

# Write docs to s3
write_to_s3 $BUCKET $LOCAL_DOC_PATH $S3_PATH

