#!/usr/bin/env bash

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

index_update_str = ''
if [[ -z $PR_NUMBER ]]
then
    bucket=''
    path=$BRANCH/$COMMIT_SHA
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
    bucket=''
    site=$bucket/$path
    if [[ $BRANCH == 'master' ]]; then flags=''; else flags=--delete; fi
    cacheControl='--cache-control max-age=7200'
fi

other_doc_version_text='Stable Version Documentation'
other_doc_version_branch='stable'
if [[ $BRANCH == 'stable' ]] {
    other_doc_version_text='Dev Version Documentation'
    other_doc_version_branch='dev'
}
escaped_context_root="${site//\\\\\//\\\\\\\\\/}"  # replace \\/ with \\\\/

mkdir -p docs/_build/rst/tutorials/
aws s3 cp s3://autogluon-dev/build_docs/$PR_NUMBER/$COMMIT_SHA/ docs/_build/rst/tutorials/

# TODO: update index_update_str
if [[ $BRANCH == 'master' ]]
then