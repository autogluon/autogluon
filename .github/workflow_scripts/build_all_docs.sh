#!/usr/bin/env bash

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

index_update_str = ''
if [[ -z $PR_NUMBER ]]; then
    bucket = ''
    path = $BRANCH/$COMMIT_SHA
fi