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
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/unit-tests/tiny_isic2017.zip"
    download_dir = "./tiny_isic2017"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_isic2017")

    return data_dir


def get_file_df(need_test_gt=False):
    data_dir = download_sample_dataset()
    train_img_files = get_file_paths(os.path.join(data_dir, "train/ISIC-2017_Train"))
    train_gt_files = get_file_paths(os.path.join(data_dir, "train/ISIC-2017_Training_Part1_GroundTruth"))
    val_img_files = get_file_paths(os.path.join(data_dir, "val/ISIC-2017_Val"))
    val_gt_files = get_file_paths(os.path.join(data_dir, "val/ISIC-2017_Validation_Part1_GroundTruth"))
    test_img_files = get_file_paths(os.path.join(data_dir, "test/ISIC-2017_Test"))
    test_gt_files = get_file_paths(os.path.join(data_dir, "test/ISIC-2017_Test_v2_Part1_GroundTruth"))

    train_df = pd.DataFrame({"image": train_img_files, "label": train_gt_files})
    val_df = pd.DataFrame({"image": val_img_files, "label": val_gt_files})
    if need_test_gt:
        test_df = pd.DataFrame({"image": test_img_files, "label": test_gt_files})
    else:
        test_df = pd.DataFrame({"image": test_img_files})
    return train_df, val_df, test_df


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "facebook/sam-vit-base",
    ],
)
def test_sam_semantic_segmentation_fit_evaluate_predict_isic(checkpoint_name):
    train_df, val_df, test_df = get_file_df(need_test_gt=True)

    validation_metric = "binary_iou"
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
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
def test_sam_semantic_segmentation_save_and_load(checkpoint_name):
    train_df, val_df, test_df = get_file_df(need_test_gt=False)

    validation_metric = "binary_iou"
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
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

    predictor.save("./sam_semantic_segmentation_save_and_load")
    new_predictor = MultiModalPredictor.load("./sam_semantic_segmentation_save_and_load")

    new_pred = new_predictor.predict(test_df, save_results=False)

    assert np.allclose(pred, new_pred)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "facebook/sam-vit-base",
    ],
)
def test_sam_semantic_segmentation_zero_shot_evaluate_predict(checkpoint_name):
    _, _, test_df = get_file_df(need_test_gt=True)

    validation_metric = "binary_iou"
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        validation_metric=validation_metric,
        eval_metric=validation_metric,
        hyperparameters={
            "env.num_gpus": 1,
            "model.sam.checkpoint_name": checkpoint_name,
        },
        label="label",
    )

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
def test_sam_semantic_segmentation_lora_insert(checkpoint_name):
    _, _, test_df = get_file_df(need_test_gt=True)

    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        hyperparameters={
            "env.num_gpus": 1,
            "model.sam.checkpoint_name": checkpoint_name,
        },
        label="label",
    )

    # Evaluate
    predictor.evaluate(test_df, metrics=["binary_iou"])

    model = predictor._learner._model
    if hasattr(model, "frozen_layers") and model.frozen_layers:
        for k, v in model.named_parameters():
            for filter_layer in model.frozen_layers:
                if filter_layer in k:
                    assert "lora" not in k
