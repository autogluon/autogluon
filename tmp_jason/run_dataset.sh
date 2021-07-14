#!/bin/bash
set -e

DATASET_NAME=$1
RESOURCE=$2
if [[ -z "$DATASET_NAME" ]] || [[ -z "$RESOURCE" ]]
then
    echo "Usage: ./run_dataset <DATASET NAME> <RESOURCE COUNT>"
    exit 1
fi

N1_RESOURCE=$(($RESOURCE * 2))
N1_R4_RESOURCE=$(($RESOURCE * 5))
TRAIN_PATH="dataset/${DATASET_NAME}/train_data.csv"
TEST_PATH="dataset/${DATASET_NAME}/test_data.csv"
N1_TRAIN_PATH="dataset/${DATASET_NAME}/n1_r1_train_data.csv"
N1_TEST_PATH="dataset/${DATASET_NAME}/n1_r1_test_data.csv"
N1_R4_TRAIN_PATH="dataset/${DATASET_NAME}/n1_r4_train_data.csv"
N1_R4_TEST_PATH="dataset/${DATASET_NAME}/n1_r4_test_data.csv"

python plot_test_scores.py -f ${TRAIN_PATH} -g ${TEST_PATH} -s 10 -r ${RESOURCE} -n plots/score/${DATASET_NAME} -z catboost -b 0
python plot_test_scores.py -f ${TRAIN_PATH} -g ${TEST_PATH} -s 10 -r ${RESOURCE} -n plots/score/${DATASET_NAME}_bag -z catboost -b 1
python plot_test_scores.py -f ${N1_TRAIN_PATH} -g ${N1_TEST_PATH} -s 10 -r ${N1_RESOURCE} -n plots/score/${DATASET_NAME}_n1 -z catboost -b 0
python plot_test_scores.py -f ${N1_TRAIN_PATH} -g ${N1_TEST_PATH} -s 10 -r ${N1_RESOURCE} -n plots/score/${DATASET_NAME}_n1_bag -z catboost -b 1
python plot_test_scores.py -f ${N1_R4_TRAIN_PATH} -g ${N1_R4_TEST_PATH} -s 10 -r ${N1_R4_RESOURCE} -n plots/score/${DATASET_NAME}_n1_r4 -z catboost -b 0
python plot_test_scores.py -f ${N1_R4_TRAIN_PATH} -g ${N1_R4_TEST_PATH} -s 10 -r ${N1_R4_RESOURCE} -n plots/score/${DATASET_NAME}_n1_r4_bag -z catboost -b 1
