#!/bin/bash
set -e

for dataset_name in ad al ca co ep he hi ja mi ya ye
do
  python3 example_tabular.py --dataset_name ${dataset_name} \
  --dataset_dir ./datasets/ \
  --exp_dir ./exp/${dataset_name} \
  --lr 1E-4 \
  --seed 0
done
