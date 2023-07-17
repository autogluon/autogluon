#!/usr/bin/env bash

MODULE=$1
REPOSITORY=$2
BRANCH=$3
PR_NUMBER=$4

if [ $MODULE == "tabular" ]
then
    python tabular/generate_framework.py --repository $REPOSITORY --branch $BRANCH
    aws s3 cp --recursive $(dirname "$0")/tabular/amlb_user_dir/* s3://autogluon-ci-benchmark/configs/$MODULE/$PR_NUMBER/
