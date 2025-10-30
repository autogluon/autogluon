# function test_multimodal {
#     SUB_FOLDER=$1
#     ADDITIONAL_TEST_ARGS=$2

#     source $(dirname "$0")/env_setup.sh

#     setup_build_env
#     setup_hf_model_mirror "$SUB_FOLDER"
#     # Use all available GPUs
#     unset CUDA_VISIBLE_DEVICES
#     install_local_packages "common/[tests]" "core/[all,tests]" "features/"
#     install_multimodal "[tests]"

#     cd multimodal/
#     if [ -n "$ADDITIONAL_TEST_ARGS" ]
#     then
#         python -m pytest --junitxml=results.xml --runslow "$ADDITIONAL_TEST_ARGS" tests/unittests/"$SUB_FOLDER"/
#     else
#         python -m pytest --junitxml=results.xml --runslow tests/unittests/"$SUB_FOLDER"/
#     fi
# }
