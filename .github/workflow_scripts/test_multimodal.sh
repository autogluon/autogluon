function test_multimodal {
    SUB_FOLDER=$1
    ADDITIONAL_TEST_ARGS=$2

    source $(dirname "$0")/env_setup.sh

    setup_build_env
    setup_hf_model_mirror "$SUB_FOLDER"
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    install_local_packages "common/[tests]" "core/[all,tests]" "features/"
    install_multimodal "[tests]"

    echo "Additional Test Args: $ADDITIONAL_TEST_ARGS"

    # Get GPU USAGE COUNT HERE AND IF > 1 then use Multi-GPU CI
    # On 4 GPU machines enable 1 and get count to try first

    cd multimodal/
    if [ -n "$ADDITIONAL_TEST_ARGS" ]
    then
        python3 -m pytest -vv --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests/unittests/"$SUB_FOLDER"/ --durations=0
    else
        python3 -m pytest -vv --junitxml=results.xml --runslow tests/unittests/"$SUB_FOLDER"/ --durations=0
    fi   
}
