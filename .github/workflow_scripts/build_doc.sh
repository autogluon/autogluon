function build_doc {
    DOC="$1"
    BRANCH="$2"
    GIT_REPO="$3"
    COMMIT_SHA="$4"
    PR_NUMBER="$5"  # For push events, PR_NUMBER will be empty
    SUB_DOC=$6  # Can be empty

    source $(dirname "$0")/write_to_s3.sh
    source $(dirname "$0")/setup_mmcv.sh

    setup_build_contrib_env
    bash docs/build_pip_install.sh
    setup_mmcv

    if [[ -n $PR_NUMBER ]]; then
        BUCKET=autogluon-ci
        S3_PATH=s3://$BUCKET/build_docs/$PR_NUMBER/$COMMIT_SHA/$DOC/
    else
        BUCKET=autogluon-ci-push
        S3_PATH=s3://$BUCKET/build_docs/$BRANCH/$COMMIT_SHA/$DOC/
    fi

    DOC_PATH=docs/_build/tutorials/$DOC/
    SPHINX_BUILD_TAG="$DOC"

    if [[ -n $SUB_DOC ]]; then
        DOC_PATH+="/$SUB_DOC/"
        S3_PATH+="/$SUB_DOC/"
        SPHINX_BUILD_TAG+="/$SUB_DOC"
    fi

    cd docs
    rm -rf _build
    sphinx-build -t $SPHINX_BUILD_TAG -b html . _build/

    COMMAND_EXIT_CODE=$?
    if [ $COMMAND_EXIT_CODE -ne 0 ]; then
        exit COMMAND_EXIT_CODE
    fi

    cd ..
    write_to_s3 $BUCKET $DOC_PATH $S3_PATH
}
