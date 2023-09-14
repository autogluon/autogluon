#!/usr/bin/env bash

REPOSITORY=$1
BRANCH=$(basename $2)
SHORT_SHA=$3
PR_NUMBER=$4

# generate tabular configs
python $(dirname "$0")/tabular/generate_framework.py --repository https://github.com/$REPOSITORY.git --branch $BRANCH
if [ -n "$PR_NUMBER" ]
then
    CONFIG_PATH=tabular/$PR_NUMBER
else
    CONFIG_PATH=tabular/$BRANCH
fi

# keep commit sha for future reference
aws s3 cp --recursive $(dirname "$0")/tabular/amlb_user_dir/ s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/$SHORT_SHA/
aws s3 rm --recursive s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/latest/
aws s3 cp --recursive $(dirname "$0")/tabular/amlb_user_dir/ s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/latest/
