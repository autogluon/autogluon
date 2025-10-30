# function build_doc {
#     DOC="$1"
#     BRANCH="$2"
#     GIT_REPO="$3"
#     COMMIT_SHA="$4"
#     PR_NUMBER="$5"  # For push events, PR_NUMBER will be empty
#     SUB_DOC=$6  # Can be empty

#     source $(dirname "$0")/write_to_s3.sh
#     source $(dirname "$0")/setup_mmcv.sh

#     setup_build_contrib_env
#     bash docs/build_pip_install.sh
#     setup_mmcv

#     if [[ -n $PR_NUMBER ]]; then
#         BUCKET="autogluon-ci"
#         S3_PATH="s3://$BUCKET/build_docs/$PR_NUMBER/$COMMIT_SHA"
#     else
#         BUCKET="autogluon-ci-push"
#         S3_PATH="s3://$BUCKET/build_docs/$BRANCH/$COMMIT_SHA"
#     fi

#     S3_DOC_PATH="$S3_PATH/tutorials/$DOC"
#     S3_IMG_PATH="$S3_PATH/_images"

#     BUILD_DIR="_build/html"

#     LOCAL_DOC_PATH="$BUILD_DIR/tutorials/$DOC"
#     LOCAL_IMG_PATH="$BUILD_DIR/_images"

#     SPHINX_BUILD_TAG="$DOC"

#     if [[ -n $SUB_DOC ]]; then
#         LOCAL_DOC_PATH+="/$SUB_DOC"
#         S3_DOC_PATH+="/$SUB_DOC"
#         SPHINX_BUILD_TAG+="/$SUB_DOC"
#     fi

#     cd docs
#     rm -rf "$BUILD_DIR"
#     sphinx-build -t "$SPHINX_BUILD_TAG" -b html . "$BUILD_DIR"

#     COMMAND_EXIT_CODE=$?
#     if [ $COMMAND_EXIT_CODE -ne 0 ]; then
#         exit COMMAND_EXIT_CODE
#     fi

#     rm -rf "$BUILD_DIR/.doctrees/" # remove build artifacts that are not needed to serve webpage

#     write_to_s3 $BUCKET $LOCAL_DOC_PATH $S3_DOC_PATH
#     write_to_s3 $BUCKET $LOCAL_IMG_PATH $S3_IMG_PATH

#     cd ..
# }
