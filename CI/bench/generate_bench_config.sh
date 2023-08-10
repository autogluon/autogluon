#!/usr/bin/env bash

MODULE=$1
PRESET=$2
BENCHMARK=$3
TIME_LIMIT=$4
USER_DIR_S3_PREFIX=$5  # where to find the pre-generated config. This will either be a branch name or a PR number

CDK_DEPLOY_ACCOUNT=369469875935
CDK_DEPLOY_REGION=us-east-1
METRICS_BUCKET=autogluon-ci-benchmark
FRAMEWORK=AutoGluon_$PRESET:benchmark

if [ $MODULE == "tabular" ]
then
    INSTANCE_TYPE=m5.2xlarge
    aws s3 cp --recursive s3://autogluon-ci-benchmark/configs/tabular/$USER_DIR_S3_PREFIX/latest/ $(dirname "$0")/amlb_user_dir/
fi

agbench generate-cloud-config \
--prefix ag-bench-${INSTANCE_TYPE//./} \
--module $MODULE \
--cdk-deploy-account $CDK_DEPLOY_ACCOUNT \
--cdk-deploy-region $CDK_DEPLOY_REGION \
--metrics-bucket $METRICS_BUCKET \
--instance $INSTANCE_TYPE \
--framework $FRAMEWORK \
--amlb-benchmark $BENCHMARK \
--amlb-constraint $TIME_LIMIT \
--amlb-user-dir $(dirname "$0")/amlb_user_dir \
--git-uri-branch https://github.com/openml/automlbenchmark.git#stable
