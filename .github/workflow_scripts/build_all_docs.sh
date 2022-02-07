#!/usr/bin/env bash

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

set -ex

source $(dirname "$0")/env_setup.sh

if [[ -z $PR_NUMBER ]]
then
    bucket='autogluon-ci'
    path=staging/$BRANCH/$COMMIT_SHA
    site=$bucket.s3-website-us-region.amazonaws.com/$path
    flags='--delete'
    cacheControl=''
else
    if [[ $BRANCH == 'master' ]]
    then
        path='dev'
    else
        if [[ $BRANCH == 'dev' ]]
        then
            path='dev-branch'
        else
            path=$BRANCH
        fi
    fi
    bucket='autogluon-ci'  # TODO: update this to real bucket
    site=$bucket/$path
    if [[ $BRANCH == 'master' ]]; then flags=''; else flags=--delete; fi
    cacheControl='--cache-control max-age=7200'
fi

other_doc_version_text='Stable Version Documentation'
other_doc_version_branch='stable'
if [[ $BRANCH == 'stable' ]]
then
    other_doc_version_text='Dev Version Documentation'
    other_doc_version_branch='dev'
fi
escaped_context_root="${site//\\\\\//\\\\\\\\\/}"  # replace \\/ with \\\\/

mkdir -p docs/_build/rst/tutorials/
# aws s3 cp s3://autogluon-ci/build_docs/$PR_NUMBER/$COMMIT_SHA/ docs/_build/rst/tutorials/ --recursive
aws s3 cp s3://autogluon-ci/build_docs/master/ee10b2d/ docs/_build/rst/tutorials/ --recursive  # test

setup_build_contrib_env
install_all
setup_mxnet_gpu
setup_torch

sed -i -e 's/###_PLACEHOLDER_WEB_CONTENT_ROOT_###/http:\\/\\/${escaped_context_root}/g' docs/config.ini
sed -i -e 's/###_OTHER_VERSIONS_DOCUMENTATION_LABEL_###/${other_doc_version_text}/g' docs/config.ini
sed -i -e 's/###_OTHER_VERSIONS_DOCUMENTATION_BRANCH_###/${other_doc_version_branch}/g' docs/config.ini

shopt -s extglob
cd docs && d2lbook build rst && d2lbook build html

COMMAND_EXIT_CODE=$?
if [ $COMMAND_EXIT_CODE -ne 0 ]; then
    exit COMMAND_EXIT_CODE
fi

aws s3 sync ${flags} _build/html/ s3://${bucket}/${path} --acl public-read ${cacheControl}
echo "Uploaded doc to http://${site}/index.html"

if [[ $BRANCH == 'master' ]]
then
    aws s3 cp root_index.html s3://${bucket}/index.html --acl public-read ${cacheControl}
    echo "Uploaded root_index.html s3://${bucket}/index.html"
fi
