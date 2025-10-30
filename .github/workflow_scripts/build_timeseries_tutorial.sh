# #!/bin/bash

# set -ex
# shopt -s extglob

# BRANCH=$(basename $1)
# GIT_REPO=$2
# COMMIT_SHA=$3
# PR_NUMBER=$4  # For push events, PR_NUMBER will be empty

# source $(dirname "$0")/env_setup.sh
# source $(dirname "$0")/build_doc.sh

# export CUDA_VISIBLE_DEVICES=0

# build_doc timeseries $BRANCH $GIT_REPO $COMMIT_SHA $PR_NUMBER
