#!/usr/bin/env bash

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

set -ex

source $(dirname "$0")/env_setup.sh

if [[ -n $PR_NUMBER ]]; then build_docs_path=build_docs/$PR_NUMBER/$COMMIT_SHA; else build_docs_path=build_docs/$BRANCH/$COMMIT_SHA; fi

if [[ (-n $PR_NUMBER) || ($GIT_REPO != awslabs/autogluon) ]]
then
    bucket='autogluon-doc-staging'
    if [[ -n $PR_NUMBER ]]; then path=staging/$PR_NUMBER/$COMMIT_SHA; else path=staging/$BRANCH/$COMMIT_SHA; fi
    site=$bucket.s3-website-us-region.amazonaws.com/$path
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
    bucket='autogluon-website'
    site=$bucket/$path
    if [[ $BRANCH == 'master' ]]; then flags=''; else flags=--delete; fi
    cacheControl='--cache-control max-age=7200'
fi

COMMAND_EXIT_CODE=$?
if [ $COMMAND_EXIT_CODE -ne 0 ]; then
    exit COMMAND_EXIT_CODE
fi

aws s3 sync ${flags} s3://autogluon-ci/${build_docs_path}/all s3://${bucket}/${path} --acl public-read ${cacheControl}
echo "Uploaded doc to http://${site}/index.html"

if [[ ($BRANCH == 'master') && ($REPO == awslabs/autogluon) ]]
then
    aws s3 cp s3://autogluon-ci/${build_docs_path}/root_index.html s3://${bucket}/index.html --acl public-read ${cacheControl}
    echo "Uploaded root_index.html s3://${bucket}/index.html"
fi
