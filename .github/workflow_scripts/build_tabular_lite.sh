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
    LITE_DIR="$BUILD_DIR/_lite"
    WHEEL_DIR="$LITE_DIR/pypi"
    FILES_DIR="$LITE_DIR/files"
    mkdir -p $WHEEL_DIR $FILES_DIR
    LITE_DIR=$(realpath $LITE_DIR)
    export WHEEL_DIR=$(realpath $WHEEL_DIR)
    build_tabular_lite

    CONTENTS_DIR=$(pwd)/docs/tutorials/tabular
    SUPPORTED_NOTEBOOKS="tabular-quick-start.ipynb tabular-indepth.ipynb tabular-feature-engineering.ipynb advanced/tabular-custom-metric.ipynb advanced/tabular-custom-model-advanced.ipynb advanced/tabular-custom-model.ipynb advanced/tabular-multilabel.ipynb"
    for nb in $SUPPORTED_NOTEBOOKS
    do
        cp ${CONTENTS_DIR}/$nb ${FILES_DIR}
    done

    mkdir -p $LOCAL_IMG_PATH
    cd $LOCAL_IMG_PATH
    jupyter lite build --output-dir dist --lite-dir $LITE_DIR
    rm -f .jupyterlite.doit.db
    cd -

    COMMAND_EXIT_CODE=$?
    if [ $COMMAND_EXIT_CODE -ne 0 ]; then
        exit COMMAND_EXIT_CODE
    fi

    write_to_s3 $BUCKET $LOCAL_IMG_PATH $S3_IMG_PATH
}
