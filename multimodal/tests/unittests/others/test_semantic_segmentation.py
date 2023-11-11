import os

import numpy as np
import pandas as pd
import pytest
import torch

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor


# modify the load_sem_seg function in detectron2
def file2id(folder_path, file_path, split_str="_"):
    image_id = os.path.normpath(os.path.relpath(file_path, start=folder_path))
    if split_str in image_id:
        image_id = os.path.splitext(image_id)[0].split(split_str)[0]
    else:
        image_id = os.path.splitext(image_id)[0]
    return image_id


def get_file_paths(directory, split_str="_"):
    file_paths = sorted(os.listdir(directory), key=lambda file_path: file2id(directory, file_path, split_str))
    return [os.path.join(directory, file_path) for file_path in file_paths]


def download_binary_semantic_seg_sample_dataset():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/unit-tests/tiny_isic2017.zip"
    download_dir = "./tiny_isic2017"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_isic2017")

    return data_dir


def download_multi_semantic_seg_sample_dataset():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/unit-tests/tiny_trans10kcls12.zip"
    download_dir = "./tiny_trans10kcls12"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_trans10kcls12")

    return data_dir


def get_file_df_binary_semantic_seg(need_test_gt=False):
    data_dir = download_binary_semantic_seg_sample_dataset()
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


def get_file_df_multi_semantic_seg(need_test_gt=False):
    data_dir = download_multi_semantic_seg_sample_dataset()
    train_img_files = get_file_paths(os.path.join(data_dir, "train/images"))
    train_gt_files = get_file_paths(os.path.join(data_dir, "train/masks_12"))
    val_img_files = get_file_paths(os.path.join(data_dir, "validation/images"))
    val_gt_files = get_file_paths(os.path.join(data_dir, "validation/masks_12"))
    test_img_files = get_file_paths(os.path.join(data_dir, "test/images"))
    test_gt_files = get_file_paths(os.path.join(data_dir, "test/masks_12"))

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
    train_df, val_df, test_df = get_file_df_binary_semantic_seg(need_test_gt=True)

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
    train_df, val_df, test_df = get_file_df_binary_semantic_seg(need_test_gt=False)

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
    _, _, test_df = get_file_df_binary_semantic_seg(need_test_gt=True)

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
def test_sam_semantic_segmentation_fit_evaluate_predict_trans10k(checkpoint_name):
    train_df, val_df, test_df = get_file_df_multi_semantic_seg(need_test_gt=True)

    validation_metric = "multiclass_iou"
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        validation_metric=validation_metric,
        eval_metric=validation_metric,
        hyperparameters={
            "env.num_gpus": 1,
            "env.precision": 32,
            "model.sam.checkpoint_name": checkpoint_name,
            "optimization.loss_function": "mask2former_loss",
            "model.sam.num_mask_tokens": 5,
        },
        label="label",
        num_classes=12,
    )

    predictor.fit(train_data=train_df, tuning_data=val_df, time_limit=20)

    # Evaluate
    predictor.evaluate(test_df, metrics=[validation_metric])

    # Predict
    predictor.predict(test_df, save_results=False)


@pytest.mark.parametrize(
    "frozen_layers",
    [
        ["prompt_encoder"],
        ["vision_encoder"],
    ],
)
def test_sam_semantic_segmentation_lora_insert(frozen_layers):
    _, _, test_df = get_file_df(need_test_gt=True)
    # SAM's vision encoder has query and value linear layers, while the prompt encoder does not.
    predictor = MultiModalPredictor(
        problem_type="semantic_segmentation",
        hyperparameters={
            "env.num_gpus": 1,
            "model.sam.checkpoint_name": "facebook/sam-vit-base",
            "model.sam.frozen_layers": frozen_layers,
        },
        label="label",
    )
    # Evaluate
    predictor.evaluate(test_df, metrics=["binary_iou"])
    model = predictor._learner._model
    assert hasattr(model, "frozen_layers") and len(model.frozen_layers) > 0
    for k, v in model.named_parameters():
        for filter_layer in model.frozen_layers:
            if filter_layer in k:
                assert "lora" not in k
