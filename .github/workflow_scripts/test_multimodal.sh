function test_multimodal {
    SUB_FOLDER=$1
    ADDITIONAL_TEST_ARGS=$2

    source $(dirname "$0")/env_setup.sh

    setup_build_env
    setup_torch_gpu
    setup_hf_model_mirror "$SUB_FOLDER"
    export CUDA_VISIBLE_DEVICES=0
    install_local_packages "common/[tests]" "core/[all,tests]" "features/"
    install_multimodal "[tests]"

    cd multimodal/
    if [ -n "$ADDITIONAL_TEST_ARGS" ]
    then
        python3 -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests/unittests/"$SUB_FOLDER"/
    else
        python3 -m pytest --junitxml=results.xml --runslow tests/unittests/"$SUB_FOLDER"/
    fi   
}
