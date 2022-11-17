function build_doc {
    DOC="$1"
    BRANCH="$2"
    GIT_REPO="$3"
    COMMIT_SHA="$4"
    PR_NUMBER="$5"  # For push events, PR_NUMBER will be empty

    source $(dirname "$0")/write_to_s3.sh
    source $(dirname "$0")/setup_mmcv.sh

    setup_build_contrib_env
    bash docs/build_pip_install.sh
    setup_mmcv
    # only build for docs/$DOC
    rm -rf ./docs/tutorials/!($DOC)
    cd docs && rm -rf _build && rm -rf juptyer_execute/ && sphinx-build -b html . _build/

    COMMAND_EXIT_CODE=$?
    if [ $COMMAND_EXIT_CODE -ne 0 ]; then
        exit COMMAND_EXIT_CODE
    fi

    cd ..

    if [[ -n $PR_NUMBER ]]; then BUCKET=autogluon-ci S3_PATH=s3://$BUCKET/build_docs/$PR_NUMBER/$COMMIT_SHA; else BUCKET=autogluon-ci-push S3_PATH=s3://$BUCKET/build_docs/$BRANCH/$COMMIT_SHA; fi
    DOC_PATH=docs/_build/tutorials/$DOC/
    S3_PATH=$S3_PATH/$DOC/

    write_to_s3 $BUCKET $DOC_PATH $S3_PATH
}
