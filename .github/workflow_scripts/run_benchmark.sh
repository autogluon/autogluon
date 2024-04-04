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
echo "----Copying and Printing Cloud Configs----"
cat $MODULE"_cloud_configs.yaml"
agbench run $MODULE"_cloud_configs.yaml" --wait

# If it is a PR, fetch the cleaned file of master-evaluation
if [ $BRANCH_OR_PR_NUMBER != "master" ]; then
    # Capture the name of the file, rename it and store it in ./results
    if [ $MODULE != "multimodal" ]; then
        master_cleaned_file=$(aws s3 ls s3://autogluon-ci-benchmark/cleaned/$MODULE/master/latest/ | awk '{print $NF}')
        new_master_cleaned_file="master_${master_cleaned_file}"
        aws s3 cp --recursive s3://autogluon-ci-benchmark/cleaned/$MODULE/master/latest/ ./results
        mv "./results/$master_cleaned_file" "./results/$new_master_cleaned_file"
    else
        master_cleaned_file=$(aws s3 ls s3://autogluon-ci-benchmark/cleaned/$MODULE/$BENCHMARK/master/latest/ | awk '{print $NF}')
        new_master_cleaned_file="master_${master_cleaned_file}"
        aws s3 cp --recursive s3://autogluon-ci-benchmark/cleaned/$MODULE/$BENCHMARK/master/latest/ ./results
        mv "./results/$master_cleaned_file" "./results/$new_master_cleaned_file"
    fi
fi

python CI/bench/evaluate.py --config_path ./ag_bench_runs/$MODULE/ --module_name $MODULE --time_limit $TIME_LIMIT --branch_name $BRANCH_OR_PR_NUMBER --benchmark_type $BENCHMARK

echo "Deleting version1.0_file"
rm -f ./results/version1.0*

for file in ./results/*; do
    CLEANED_PATH="s3://autogluon-ci-benchmark/cleaned/$MODULE"
    EVALUATION_PATH="s3://autogluon-ci-benchmark/evaluation/$MODULE"
    BRANCH_NAME="master"
    if [[ "$(basename "$file")" != "master"* ]]; then
        BRANCH_NAME="$BRANCH_OR_PR_NUMBER"
    fi

    if [ $MODULE == "multimodal" ]; then
        CLEANED_PATH="$CLEANED_PATH/$BENCHMARK"
        EVALUATION_PATH="$EVALUATION_PATH/$BENCHMARK"
    fi
 
    aws s3 cp "$file" "$CLEANED_PATH/$BRANCH_NAME/$SHA/$(basename "$file")"
    aws s3 rm --recursive "$CLEANED_PATH/$BRANCH_NAME/latest/"
    aws s3 cp "$file" "$CLEANED_PATH/$BRANCH_NAME/latest/$(basename "$file")"

    if [[ "$(basename "$file")" == "master"* ]]; then
        aws s3 cp --recursive ./evaluate "$EVALUATION_PATH/$BRANCH_NAME/$SHA/"
        aws s3 rm --recursive "$EVALUATION_PATH/$BRANCH_NAME/latest/"
        aws s3 cp --recursive ./evaluate "$EVALUATION_PATH/$BRANCH_NAME/latest/"
    fi
done

# Run dashboard if the branch is not master
if [ $BRANCH_OR_PR_NUMBER != "master" ]
then
    echo "Name of all files"
    ls
    cwd=`pwd`
    ls ./evaluate/pairwise/* | grep .csv > $cwd/agg_csv.txt
    cat agg_csv.txt
    filename=`head -1 $cwd/agg_csv.txt`
    prefix=$BRANCH_OR_PR_NUMBER/$SHA
    agdash --per_dataset_csv  './evaluate/results_ranked_by_dataset_all.csv' --agg_dataset_csv $filename --s3_prefix benchmark-dashboard/$prefix --s3_bucket autogluon-staging --s3_region us-west-2 > $cwd/out.txt
    tail -1 $cwd/out.txt > $cwd/website.txt
fi
