#!/bin/bash

set -ex

MODULE=$1
PRESET=$2
BENCHMARK=$3
TIME_LIMIT=$4
BRANCH_OR_PR_NUMBER=$5
SHA=$6

source $(dirname "$0")/env_setup.sh

setup_benchmark_env

/bin/bash CI/bench/generate_bench_config.sh $MODULE $PRESET $BENCHMARK $TIME_LIMIT $BRANCH_OR_PR_NUMBER
agbench run $MODULE"_cloud_configs.yaml" --wait

python CI/bench/evaluate.py --config_path ./ag_bench_runs/tabular/ --time_limit $TIME_LIMIT
aws s3 cp --recursive ./results s3://autogluon-ci-benchmark/cleaned/$BRANCH_OR_PR_NUMBER/$SHA/
aws s3 rm --recursive s3://autogluon-ci-benchmark/cleaned/$BRANCH_OR_PR_NUMBER/latest/
aws s3 cp --recursive ./results s3://autogluon-ci-benchmark/cleaned/$BRANCH_OR_PR_NUMBER/latest/
