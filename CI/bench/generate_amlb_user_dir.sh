#!/usr/bin/env bash

REPOSITORY=$1
BRANCH=$(basename $2)
PR_NUMBER=$3

# generate tabular configs
python $(dirname "$0")/tabular/generate_framework.py --repository $REPOSITORY --branch $BRANCH
if [ -n "$PR_NUMBER" ]
then
    CONFIG_PATH=tabular/$PR_NUMBER/
else
    CONFIG_PATH=tabular/$BRANCH/
fi
aws s3 cp --recursive $(dirname "$0")/tabular/amlb_user_dir/ s3://autogluon-ci-benchmark/configs/$CONFIG_PATH
