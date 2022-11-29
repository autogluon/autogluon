#!/bin/bash

set -ex
shopt -s extglob

SUB_DOC=$1
BRANCH=$(basename $2)
GIT_REPO=$3
COMMIT_SHA=$4
PR_NUMBER=$5  # For push events, PR_NUMBER will be empty

source $(dirname "$0")/env_setup.sh
source $(dirname "$0")/build_doc.sh

build_doc multimodal $SUB_DOC $BRANCH $GIT_REPO $COMMIT_SHA ${PR_NUMBER:-""} $SUB_DOC