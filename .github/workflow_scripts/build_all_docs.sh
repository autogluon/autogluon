#!/usr/bin/env bash

BRANCH=$(basename $1)
GIT_REPO=$2
COMMIT_SHA=$3
PR_NUMBER=$4

source $(dirname "$0")/env_setup.sh

index_update_str = ''
if [[ -z $PR_NUMBER ]]
then
    bucket=''
    path=$BRANCH/$COMMIT_SHA
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
    bucket=''
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
aws s3 cp s3://autogluon-ci/build_docs/$PR_NUMBER/$COMMIT_SHA/ docs/_build/rst/tutorials/

install_all
setup_mxnet_gpu
setup_torch

sed -i -e 's/###_PLACEHOLDER_WEB_CONTENT_ROOT_###/http:\\/\\/${escaped_context_root}/g' docs/config.ini
sed -i -e 's/###_OTHER_VERSIONS_DOCUMENTATION_LABEL_###/${other_doc_version_text}/g' docs/config.ini
sed -i -e 's/###_OTHER_VERSIONS_DOCUMENTATION_BRANCH_###/${other_doc_version_branch}/g' docs/config.ini

shopt -s extglob
rm -rf ./docs/tutorials/!(index.rst)
cd docs && d2lbook build rst && d2lbook build html
aws s3 sync ${flags} _build/html/ s3://${bucket}/${path} --acl public-read ${cacheControl}
echo "Uploaded doc to http://${site}/index.html"

# TODO: update index_update_str
if [[ $BRANCH == 'master' ]]
then
    aws s3 cp root_index.html s3://${bucket}/index.html --acl public-read ${cacheControl}
    echo "Uploaded root_index.html s3://${bucket}/index.html"
fi