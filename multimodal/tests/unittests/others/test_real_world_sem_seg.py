import os

import numpy as np
import pytest

from autogluon.core.utils.loaders import load_zip
from autogluon.multimodal import MultiModalPredictor


def download_sample_dataset():
    zip_file = "https://automl-mm-bench.s3.amazonaws.com/real_world_sem_seg/tiny_isic2017.zip"
    download_dir = "./tiny_isic2017"

    load_zip.unzip(zip_file, unzip_dir=download_dir)
    data_dir = os.path.join(download_dir, "tiny_isic2017")

    return data_dir


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "facebook/sam-vit-huge",
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
    )

    predictor.fit(
        train_data=f"{train_img_path}:{train_gt_path}", tuning_data=f"{val_img_path}:{val_gt_path}", time_limit=20
    )

    # Evaluate
    predictor.evaluate(f"{test_img_path}:{test_gt_path}", metrics=["binary_iou"])

    # Predict
    predictor.predict(f"{test_img_path}:{test_gt_path}", save_results=False)


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
    test_gt_path = os.path.join(test_data_dir, "ISIC-2017_Test_v2_Part1_GroundTruth")

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
    )

    predictor.fit(
        train_data=f"{train_img_path}:{train_gt_path}", tuning_data=f"{val_img_path}:{val_gt_path}", time_limit=20
    )

    # Predict
    pred = predictor.predict(f"{test_img_path}:{test_gt_path}", save_results=False)

    predictor.save("./")
    new_predictor = MultiModalPredictor.load("./")

    new_pred = new_predictor.predict(f"{test_img_path}:{test_gt_path}", save_results=False)

    assert np.allclose(pred["logits"][0], new_pred["logits"][0])
