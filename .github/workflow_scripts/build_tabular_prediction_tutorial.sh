#!/bin/bash

set -ex
shopt -s extglob

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4  # For push events, PR_NUMBER will be empty

source $(dirname "$0")/env_setup.sh
source $(dirname "$0")/build_doc.sh

export CUDA_VISIBLE_DEVICES=0

# python3 -m pip3 install --upgrade pip
# python3 -m pip install --upgrade awscli
# python3 -m pip3 install awscli==1.18.105
# python3 -m pip3 install botocore==1.17.28


build_doc tabular $BRANCH $GIT_REPO $COMMIT_SHA $PR_NUMBER
