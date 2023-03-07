#!/usr/bin/env bash

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

set -ex

source $(dirname "$0")/env_setup.sh
source $(dirname "$0")/write_to_s3.sh

if [[ -n $PR_NUMBER ]]; then BUILD_DOCS_PATH=s3://autogluon-ci/build_docs/$PR_NUMBER/$COMMIT_SHA; else BUILD_DOCS_PATH=s3://autogluon-ci-push/build_docs/$BRANCH/$COMMIT_SHA; fi

if [[ (-n $PR_NUMBER) || ($GIT_REPO != autogluon/autogluon) ]]
then
    bucket='autogluon-staging'
    if [[ -n $PR_NUMBER ]]; then path=$PR_NUMBER/$COMMIT_SHA; else path=$BRANCH/$COMMIT_SHA; fi
    site=$bucket.s3-website-us-west-2.amazonaws.com/$path
    flags='--delete'
    cacheControl=''
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
    site=$bucket/$path
    if [[ $BRANCH == 'master' ]]; then flags=''; else flags='--delete'; fi
    cacheControl='--cache-control max-age=7200'
fi

COMMAND_EXIT_CODE=$?
if [ $COMMAND_EXIT_CODE -ne 0 ]; then
    exit COMMAND_EXIT_CODE
fi

aws s3 sync ${flags} ${BUILD_DOCS_PATH}/all/ s3://${bucket}/${path} --acl public-read ${cacheControl}
echo "Uploaded doc to http://${site}/index.html"

