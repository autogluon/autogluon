#!/bin/bash
set -e

DATASETS=[
"ad"
"al"
"ca"
"co"
"ep"
"he"
"hi"
"ja"
"mi"
"ya"
"ye"
]


for dataset_name in ${DATASETS}
do
  python3 example_tabular.py --dataset_name ${dataset_name} --dataset_dir ./dataset/${dataset_name} --exp_dir ./result/${dataset_name} --lr 5E-4 --seed 0
done
