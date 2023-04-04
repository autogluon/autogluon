#!/bin/bash

function build_tabular_lite {
    export AUTOGLUON_PACKAGE_NAME="autogluon-lite"

    setup_build_env
    install_local_packages "common/[tests]" "core/[all,tests]" "features/" "tabular/[all,tests]"
    build_pkg "common" "core" "features" "tabular" "autogluon"
}

function build_doc_lite {
    BRANCH="$1"
    GIT_REPO="$2"
    COMMIT_SHA="$3"
    PR_NUMBER="$4"  # For push events, PR_NUMBER will be empty

    DOC="tabular"
    SUB_DOC="lite"

    source $(dirname "$0")/write_to_s3.sh

    setup_build_contrib_env
    setup_build_jupyterlite_env

    bash docs/build_pip_install.sh

    if [[ -n $PR_NUMBER ]]; then
        BUCKET="autogluon-ci"
        S3_PATH="s3://$BUCKET/build_docs/$PR_NUMBER/$COMMIT_SHA"
    else
        BUCKET="autogluon-ci-push"
        S3_PATH="s3://$BUCKET/build_docs/$BRANCH/$COMMIT_SHA"
    fi

    S3_IMG_PATH="$S3_PATH/_images"

    BUILD_DIR="_build/html"

    LOCAL_IMG_PATH="$BUILD_DIR/_images"

    rm -rf "$BUILD_DIR"
    export WHEEL_DIR="$LOCAL_IMG_PATH/wheel"
    build_tabular_lite

    CONTENTS_DIR=$(pwd)/docs/tutorials/tabular
    mkdir -p $LOCAL_IMG_PATH
    cd $LOCAL_IMG_PATH
    jupyter lite build --content $CONTENTS_DIR --output-dir dist
    cd -

    COMMAND_EXIT_CODE=$?
    if [ $COMMAND_EXIT_CODE -ne 0 ]; then
        exit COMMAND_EXIT_CODE
    fi

    write_to_s3 $BUCKET $LOCAL_IMG_PATH $S3_IMG_PATH
}
