function test_multimodal {
    SUB_FOLDER=$1
    ADDITIONAL_TEST_ARGS=$2

    source $(dirname "$0")/env_setup.sh

    setup_build_env
    setup_hf_model_mirror "$SUB_FOLDER"

    # Install system dependencies for PDF and OCR (needed for LayoutLMv3 tests)
    if command -v apt-get &> /dev/null; then
        echo "Installing poppler-utils and tesseract-ocr..."
        apt-get update
        apt-get install -y poppler-utils tesseract-ocr libtesseract-dev
    fi

    # Use all available GPUs
    unset CUDA_VISIBLE_DEVICES
    install_local_packages "common/[tests]" "features/" "core/[all,tests]"
    install_multimodal "[tests]"

    cd multimodal/
    if [ -n "$ADDITIONAL_TEST_ARGS" ]
    then
        python -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests/unittests/"$SUB_FOLDER"/
    else
        python -m pytest --junitxml=results.xml --runslow tests/unittests/"$SUB_FOLDER"/
    fi   
}
