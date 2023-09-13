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
#copy the generated config file to test
echo "Printing the cloud config file"
cat $MODULE"_cloud_configs.yaml"
agbench run $MODULE"_cloud_configs.yaml" --wait

# If PR, fetch the cleaned file from master location here 
if [ $BRANCH_OR_PR_NUMBER != "master" ]
then
    #capture the name of the file, remane it and store it in ./results
    master_cleaned_file=$(aws s3 ls s3://autogluon-ci-benchmark/cleaned/master/latest/ | awk '{print $NF}')
    new_master_cleaned_file="master_${master_cleaned_file}"
    aws s3 cp --recursive s3://autogluon-ci-benchmark/cleaned/master/latest/ ./results
    mv "./results/$master_cleaned_file" "./results/$new_master_cleaned_file"
fi

echo "Printing AG Bench Runs"
ls ./ag_bench_runs/tabular/
python CI/bench/evaluate.py --config_path ./ag_bench_runs/tabular/ --time_limit $TIME_LIMIT --branch_name $BRANCH_OR_PR_NUMBER

for file in ./results/*; do
    echo "File Name: $file"
    # Check if the file does not start with "master"
    if [[ "$(basename "$file")" != "master"* ]]
    then
        aws s3 cp "$file" "s3://autogluon-ci-benchmark/cleaned/$BRANCH_OR_PR_NUMBER/$SHA/$(basename "$file")"
        aws s3 rm --recursive s3://autogluon-ci-benchmark/cleaned/$BRANCH_OR_PR_NUMBER/latest/
        aws s3 cp "$file" s3://autogluon-ci-benchmark/cleaned/$BRANCH_OR_PR_NUMBER/latest/$(basename "$file")
    else
        aws s3 cp "$file" "s3://autogluon-ci-benchmark/cleaned/master/$SHA/$(basename "$file")"
        aws s3 rm --recursive s3://autogluon-ci-benchmark/cleaned/master/latest/
        aws s3 cp "$file" s3://autogluon-ci-benchmark/cleaned/master/latest/$(basename "$file")
    fi
done

#run dashboard if the branch is not master
if [ $BRANCH_OR_PR_NUMBER != "master" ]
then
    cwd=`pwd`
    echo "Printing paths and folder structure"
    ls ./results
    ls data/results/output/openml/ag_eval/pairwise/* | grep .csv > $cwd/agg_csv.txt
    echo "Printing the agg_csv file"
    cat agg_csv.txt
    filename=`head -1 $cwd/agg_csv.txt`
    prefix=$BRANCH_OR_PR_NUMBER/$SHA
    agdash --per_dataset_csv  'data/results/output/openml/ag_eval/results_ranked_by_dataset_all.csv' --agg_dataset_csv $filename --s3_prefix benchmark-dashboard/$prefix --s3_bucket autogluon-staging --s3_region us-west-2 > $cwd/out.txt
    tail -1 $cwd/out.txt > $cwd/website.txt
fi
