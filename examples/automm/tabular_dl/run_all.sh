#!/bin/bash
set -e

for dataset_name in mi ya ye # ad al ca co ep he hi ja mi 
do
  python3 examples/automm/tabular_dl/example_tabular.py --dataset_name ${dataset_name} \
  --dataset_dir ./datasets/ \
  --exp_dir ./exp/original_fttransformer/${dataset_name} \
  --lr 1E-4 \
  --seed 0
done
