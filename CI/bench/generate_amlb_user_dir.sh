#!/usr/bin/env bash

REPOSITORY=$1
BRANCH=$2
PR_NUMBER=$3

# generate tabular configs
python tabular/generate_framework.py --repository $REPOSITORY --branch $BRANCH
aws s3 cp --recursive $(dirname "$0")/tabular/amlb_user_dir/* s3://autogluon-ci-benchmark/configs/$MODULE/$PR_NUMBER/
