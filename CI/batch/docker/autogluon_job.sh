#!/bin/bash

date
echo "Args: $@"
env
echo "jobId: $AWS_BATCH_JOB_ID"
echo "jobQueue: $AWS_BATCH_JQ_NAME"
echo "computeEnvironment: $AWS_BATCH_CE_NAME"

SOURCE_REF=$1
WORK_DIR=$2
COMMAND=$3
SAVED_OUTPUT=$4
SAVE_PATH=$5
REMOTE=$6
SAFE_TO_USE_SCRIPT=$7

# Copy the workflow from master branch
git clone https://github.com/autogluon/autogluon.git
WORKFLOW_SCRIPTS=autogluon/.github/workflow_scripts
if [ -d "$WORKFLOW_SCRIPTS" ]; then
    cp -R autogluon/.github/workflow_scripts .
fi

cd autogluon

if [ ! -z $REMOTE ]; then
    git remote set-url origin $REMOTE
fi

git fetch origin $SOURCE_REF:working
git checkout working

# If not safe to use script, we overwrite with the script from master branch
TRUE=true
if [[ ${SAFE_TO_USE_SCRIPT,,} != ${TRUE,,} ]]; then
    if [ -d ../workflow_scripts ]; then
        rm -rf .github/workflow_scripts
        mv ../workflow_scripts .github/
    else
        echo Not safe to use user provided script, and could not find script from master branches
        exit 1
    fi
fi

cd $WORK_DIR
/bin/bash -o pipefail -c "eval $COMMAND"
COMMAND_EXIT_CODE=$?

exit $COMMAND_EXIT_CODE
