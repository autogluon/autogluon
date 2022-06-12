#!/bin/bash
set -e

DATASETS=(
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
)


for name in ${DATASETS}
do
  python3 example_tabular.py --dataset_name ${name} --dataset_dir ./dataset/${name} --exp_dir ./result/${name} --lr 2E-4 --seed 0
done
