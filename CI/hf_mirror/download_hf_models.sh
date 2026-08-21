#!/bin/bash

if grep -qs '/mnt/efs ' /proc/mounts;
then
    echo EFS attached
else
    echo EFS failed to attach
    exit 1
fi
git clone https://github.com/autogluon/autogluon.git
python3 download_hf_models.py \
    --model_list_file autogluon/multimodal/tests/hf_model_list.yaml \
    --dataset_list_file autogluon/multimodal/tests/hf_dataset_list.yaml
aws s3 sync --delete /mnt/efs s3://autogluon-hf-model-mirror
