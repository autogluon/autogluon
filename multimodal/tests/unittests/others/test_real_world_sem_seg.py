import os

import numpy as np
import pandas as pd
import pytest
import torch

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor


def get_file_paths(directory):
    file_paths = sorted(os.listdir(directory))
    return [os.path.join(directory, file_path) for file_path in file_paths]


def download_sample_dataset():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/real_world_sem_seg/tiny_isic2017.zip"
    download_dir = "./tiny_isic2017"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_isic2017")

    return data_dir


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "facebook/sam-vit-base",
    ],
)
def test_sam_real_world_sem_seg_fit_evaluate_predict_isic(checkpoint_name):
    data_dir = download_sample_dataset()
    train_data_dir = os.path.join(data_dir, "train")
    val_data_dir = os.path.join(data_dir, "val")
    test_data_dir = os.path.join(data_dir, "test")

    train_img_path = os.path.join(train_data_dir, "ISIC-2017_Train")
    val_img_path = os.path.join(val_data_dir, "ISIC-2017_Val")
    test_img_path = os.path.join(test_data_dir, "ISIC-2017_Test")

    train_gt_path = os.path.join(train_data_dir, "ISIC-2017_Training_Part1_GroundTruth")
    val_gt_path = os.path.join(val_data_dir, "ISIC-2017_Validation_Part1_GroundTruth")
    test_gt_path = os.path.join(test_data_dir, "ISIC-2017_Test_v2_Part1_GroundTruth")

    train_img_files = get_file_paths(train_img_path)
    train_gt_files = get_file_paths(train_gt_path)
    val_img_files = get_file_paths(val_img_path)
    val_gt_files = get_file_paths(val_gt_path)
    test_img_files = get_file_paths(test_img_path)
    test_gt_files = get_file_paths(test_gt_path)

    train_df = pd.DataFrame({"image": train_img_files, "label": train_gt_files})
    val_df = pd.DataFrame({"image": val_img_files, "label": val_gt_files})
    test_df = pd.DataFrame({"image": test_img_files, "label": test_gt_files})

    validation_metric = "binary_iou"
    predictor = MultiModalPredictor(
        problem_type="real_world_sem_seg",
        sample_data_path=train_img_path,
        validation_metric=validation_metric,
        eval_metric=validation_metric,
        hyperparameters={
            "env.num_gpus": 1,
            "model.sam.checkpoint_name": checkpoint_name,
        },
        label="label",
    )

    predictor.fit(train_data=train_df, tuning_data=val_df, time_limit=20)

    # Evaluate
    predictor.evaluate(test_df, metrics=["binary_iou"])

    # Predict
    predictor.predict(test_df, save_results=False)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "facebook/sam-vit-base",
    ],
)
def test_sam_real_world_sem_seg_save_and_load(checkpoint_name):
    data_dir = download_sample_dataset()
    train_data_dir = os.path.join(data_dir, "train")
    val_data_dir = os.path.join(data_dir, "val")
    test_data_dir = os.path.join(data_dir, "test")

    train_img_path = os.path.join(train_data_dir, "ISIC-2017_Train")
    val_img_path = os.path.join(val_data_dir, "ISIC-2017_Val")
    test_img_path = os.path.join(test_data_dir, "ISIC-2017_Test")

    train_gt_path = os.path.join(train_data_dir, "ISIC-2017_Training_Part1_GroundTruth")
    val_gt_path = os.path.join(val_data_dir, "ISIC-2017_Validation_Part1_GroundTruth")

    train_img_files = get_file_paths(train_img_path)
    train_gt_files = get_file_paths(train_gt_path)
    val_img_files = get_file_paths(val_img_path)
    val_gt_files = get_file_paths(val_gt_path)
    test_img_files = get_file_paths(test_img_path)

    train_df = pd.DataFrame({"image": train_img_files, "label": train_gt_files})
    val_df = pd.DataFrame({"image": val_img_files, "label": val_gt_files})
    test_df = pd.DataFrame({"image": test_img_files})

    validation_metric = "binary_iou"
    predictor = MultiModalPredictor(
        problem_type="real_world_sem_seg",
        sample_data_path=train_img_path,
        validation_metric=validation_metric,
        eval_metric=validation_metric,
        hyperparameters={
            "env.num_gpus": 1,
            "model.sam.checkpoint_name": checkpoint_name,
        },
        label="label",
    )

    predictor.fit(train_data=train_df, tuning_data=val_df, time_limit=20)

    # Predict
    pred = predictor.predict(test_df, save_results=False)

    predictor.save("./")
    new_predictor = MultiModalPredictor.load("./")

    new_pred = new_predictor.predict(test_df, save_results=False)

    assert torch.allclose(pred, new_pred)
