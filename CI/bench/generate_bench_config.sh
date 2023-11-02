#!/usr/bin/env bash

MODULE=$1
PRESET=$2
BENCHMARK=$3
TIME_LIMIT=$4
USER_DIR_S3_PREFIX=$5  # where to find the pre-generated config. This will either be a branch name or a PR number

CDK_DEPLOY_ACCOUNT=369469875935
CDK_DEPLOY_REGION=us-east-1
METRICS_BUCKET=autogluon-ci-benchmark

if [ $MODULE == "tabular" ] || [ $MODULE == "timeseries" ]; then
    FRAMEWORK=AutoGluon_$PRESET:benchmark
    INSTANCE_TYPE=m5.2xlarge
    aws s3 cp --recursive s3://autogluon-ci-benchmark/configs/$MODULE/$USER_DIR_S3_PREFIX/latest/ $(dirname "$0")/amlb_user_dir/
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
else
# Put out an IF here based on the BENCHMARK value prepare the dataloader
    FRAMEWORK=AutoGluon_$PRESET
    DATASET_YAML_PATH=$(dirname "$0")/custom_user_dir/dataloaders/automm_cv_datasets.yaml
    aws s3 cp --recursive s3://autogluon-ci-benchmark/configs/$MODULE/$USER_DIR_S3_PREFIX/latest/ $(dirname "$0")/custom_user_dir/

    for dataset_name in $(yq eval 'keys | .[]' "$DATASET_YAML_PATH" | grep -v '^base$')
    do
        agbench generate-cloud-config \
        --prefix ag-bench \
        --module $MODULE \
        --cdk-deploy-account $CDK_DEPLOY_ACCOUNT \
        --cdk-deploy-region $CDK_DEPLOY_REGION \
        --metrics-bucket $METRICS_BUCKET \
        --max-machine-num 200 \
        --data-bucket automl-mm-bench \
        --framework $FRAMEWORK \
        --constraint $TIME_LIMIT \
        --custom-resource-dir $(dirname "$0")/custom_user_dir \
        --dataset-names "$dataset_name" \
        --custom-dataloader "dataloader_file:$(dirname "$0")/custom_user_dir/dataloaders/vision_dataloader.py;class_name:VisionDataLoader;dataset_config_file:$(dirname "$0")/custom_user_dir/dataloaders/automm_cv_datasets.yaml"
    done
fi

# TO DO: 
# Add Max Machine Num for all 3 modules
# Store name of all the datasets
# Fill in the vision, text-tabular part
# Run Benchmark will change accordingly
# Benchmark results for multimodal will be stored in vision/text-tabular etc
