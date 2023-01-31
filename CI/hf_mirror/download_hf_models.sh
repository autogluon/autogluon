#!/bin/bash

df -h  # debug purpose to make sure efs is attached correclty
git clone https://github.com/autogluon/autogluon.git
python3 download_hf_models.py --model_list_file autogluon/multimodal/tests/hf_model_list.yaml
aws s3 sync --delete /mnt/efs s3://autogluon-hf-model-mirror
