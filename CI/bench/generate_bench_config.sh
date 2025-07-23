#!/usr/bin/env bash

MODULE=$1
PRESET=$2
BENCHMARK=$3
TIME_LIMIT=$4
USER_DIR_S3_PREFIX=$5  # where to find the pre-generated config. This will either be a branch name or a PR number

CDK_DEPLOY_ACCOUNT=369469875935
CDK_DEPLOY_REGION=us-east-1
METRICS_BUCKET=autogluon-ci-benchmark
MAX_MACHINE_NUM=1040

# Function to convert time format to seconds
convert_time_to_seconds() {
    local time_str=$1
    if [[ $time_str =~ ^([0-9]+)h$ ]]; then
        echo $(( ${BASH_REMATCH[1]} * 3600 ))
    elif [[ $time_str =~ ^([0-9]+)m$ ]]; then
        echo $(( ${BASH_REMATCH[1]} * 60 ))
    elif [[ $time_str =~ ^([0-9]+)s$ ]]; then
        echo ${BASH_REMATCH[1]}
    elif [[ $time_str =~ ^[0-9]+$ ]]; then
        echo $time_str  # Already in seconds
    else
        echo "3600"  # Default to 1 hour if format is unrecognized
    fi
}

# Convert TIME_LIMIT to seconds for AWS infrastructure timeout
TIME_LIMIT_SECONDS=$(convert_time_to_seconds "$TIME_LIMIT")

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
    --max-machine-num $MAX_MACHINE_NUM \
    --instance $INSTANCE_TYPE \
    --framework $FRAMEWORK \
    --amlb-benchmark $BENCHMARK \
    --amlb-constraint $TIME_LIMIT \
    --time-limit $TIME_LIMIT_SECONDS \
    --amlb-user-dir $(dirname "$0")/amlb_user_dir \
    --git-uri-branch https://github.com/Innixma/automlbenchmark.git#autogluon_switch_to_uv
else
    FRAMEWORK=AutoGluon_$PRESET
    aws s3 cp --recursive s3://autogluon-ci-benchmark/configs/$MODULE/$USER_DIR_S3_PREFIX/latest/ $(dirname "$0")/custom_user_dir/
    dataloader_file=""
    class_name=""
    dataset_file=""
    custom_dataloader_value=""
    custom_metrics_path=""
    custom_function_name=""
    optimum=0
    if [ $BENCHMARK == "automm-image" ]; then
        dataloader_file="vision_dataloader.py"
        class_name="VisionDataLoader"
        dataset_file="automm_cv_datasets.yaml"
        custom_dataloader_value="dataloader_file:$(dirname "$0")/custom_user_dir/dataloaders/$dataloader_file;class_name:$class_name;dataset_config_file:$(dirname "$0")/custom_user_dir/dataloaders/$dataset_file"
    elif [ $BENCHMARK == "automm-text-tabular" ]; then
        dataloader_file="text_tabular_dataloader.py"
        class_name="TextTabularDataLoader"
        dataset_file="text_tabular_datasets.yaml"
        custom_dataloader_value="dataloader_file:$(dirname "$0")/custom_user_dir/dataloaders/$dataloader_file;class_name:$class_name;dataset_config_file:$(dirname "$0")/custom_user_dir/dataloaders/$dataset_file"
    elif [ $BENCHMARK == "automm-text" ]; then
        dataloader_file="text_dataloader.py"
        class_name="TextDataLoader"
        dataset_file="text_datasets.yaml"
        custom_dataloader_value="dataloader_file:$(dirname "$0")/custom_user_dir/dataloaders/$dataloader_file;class_name:$class_name;dataset_config_file:$(dirname "$0")/custom_user_dir/dataloaders/$dataset_file"
    elif [ $BENCHMARK == "automm-text-tabular-image" ]; then
        dataloader_file="text_tabular_image_dataloader.py"
        class_name="TextTabularImageDataLoader"
        dataset_file="text_tabular_image_datasets.yaml"
        custom_dataloader_value="dataloader_file:$(dirname "$0")/custom_user_dir/dataloaders/$dataloader_file;class_name:$class_name;dataset_config_file:$(dirname "$0")/custom_user_dir/dataloaders/$dataset_file"
        custom_metrics_path="$(dirname "$0")/custom_metrics/cpp_coverage.py"
        custom_function_name="coverage"
        optimum=1
    else
        echo "Error: Unsupported benchmark '$BENCHMARK'"
        exit 1
    fi

    DATASET_YAML_PATH="$(dirname "$0")/custom_user_dir/dataloaders/$dataset_file"
    dataset_names=""
    # Use yq to extract the dataset names and concatenate them with commas
    for name in $(yq -r '. | keys[]' "$DATASET_YAML_PATH"); do
        if [ "$name" != "base" ]; then
            if [ -n "$dataset_names" ]; then
            dataset_names="$dataset_names,$name"
            else
            dataset_names="$name"
            fi
        fi
    done

    gen_bench_command="agbench generate-cloud-config \
    --prefix ag-bench \
    --module $MODULE \
    --cdk-deploy-account $CDK_DEPLOY_ACCOUNT \
    --cdk-deploy-region $CDK_DEPLOY_REGION \
    --metrics-bucket $METRICS_BUCKET \
    --max-machine-num $MAX_MACHINE_NUM \
    --data-bucket automl-mm-bench \
    --framework $FRAMEWORK \
    --constraint $TIME_LIMIT \
    --time-limit $TIME_LIMIT_SECONDS \
    --custom-resource-dir $(dirname "$0")/custom_user_dir \
    --dataset-names "$dataset_names" \
    --custom-dataloader '$custom_dataloader_value'"

    if [ $BENCHMARK == "automm-text" ]; then
        gen_bench_command="$gen_bench_command"
    elif [ $BENCHMARK == "automm-text-tabular-image" ]; then
        gen_bench_command="$gen_bench_command --custom-metrics --metrics-path $custom_metrics_path --function-name $custom_function_name --optimum $optimum --greater-is-better"
    fi
    eval "$gen_bench_command"
fi
