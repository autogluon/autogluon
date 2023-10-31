#!/usr/bin/env bash

MODULE=$1
REPOSITORY=$2
BRANCH=$3
SHORT_SHA=$4
PR_NUMBER=$5
FOLDS=$6

# Mostly does not require any change for AutoMM side, as we need to change the things in amlb-user-dir

# generate custom amlb configs
if [ -z "$FOLDS" ]; then
    python $(dirname "$0")/generate_framework.py --module $MODULE --repository https://github.com/$REPOSITORY.git --branch $BRANCH --folds_to_run -1
else
    python $(dirname "$0")/generate_framework.py --module $MODULE --repository https://github.com/$REPOSITORY.git --branch $BRANCH --folds_to_run $FOLDS
fi

if [ -n "$PR_NUMBER" ]
then
    CONFIG_PATH=$MODULE/$PR_NUMBER
else
    CONFIG_PATH=$MODULE/$BRANCH
fi

# keep commit sha for future reference
aws s3 cp --recursive $(dirname "$0")/$MODULE/amlb_user_dir/ s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/$SHORT_SHA/
aws s3 rm --recursive s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/latest/
aws s3 cp --recursive $(dirname "$0")/$MODULE/amlb_user_dir/ s3://autogluon-ci-benchmark/configs/$CONFIG_PATH/latest/
