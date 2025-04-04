import uuid

import numpy as np
import pytest
from sklearn.metrics import f1_score, log_loss

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils.misc import shopee_dataset


def test_predict_image_str_or_list():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"
    predictor = MultiModalPredictor(label="label", path=model_path)
    predictor.fit(
        train_data=train_data,
        time_limit=0,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    image_path = test_data.iloc[0]["image"]

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)


@pytest.mark.parametrize(
    "checkpoint_name",
    [
        "swin_tiny_patch4_window7_224",
    ],
)
def test_focal_loss_multiclass(checkpoint_name):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)

    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"

    predictor = MultiModalPredictor(label="label", problem_type="multiclass", path=model_path)

    predictor.fit(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": -1,
            "optimization.loss_function": "focal_loss",
            "optimization.focal_loss.alpha": [1, 0.25, 0.35, 0.16],  # shopee dataset has 4 classes.
            "optimization.focal_loss.gamma": 2.5,
            "optimization.focal_loss.reduction": "mean",
            "optimization.max_epochs": 1,
        },
        train_data=train_data,
        time_limit=30,  # seconds
    )  # you can trust the default config, e.g., we use a `swin_base_patch4_window7_224` model

    image_path = test_data.iloc[0]["image"]

    predictions_str = predictor.predict(image_path)
    predictions_list1 = predictor.predict([image_path])
    predictions_list10 = predictor.predict([image_path] * 10)


@pytest.mark.parametrize(
    "checkpoint_name,eval_metric",
    [
        ("swin_tiny_patch4_window7_224", "log_loss"),
        ("swin_tiny_patch4_window7_224", "f1_micro"),
    ],
)
def test_metrics_multiclass(checkpoint_name, eval_metric):
    """
    Test the MultiModalPredictor's evaluation metrics for multiclass classification.

    This test verifies that:
    1. The predictor correctly implements the specified evaluation metrics (log_loss and f1_micro)
    2. The manually calculated metrics match the predictor's evaluate() output
    3. The model training and prediction pipeline works end-to-end

    Parameters
    ----------
    checkpoint_name : str
        Name of the model checkpoint to use (e.g., "swin_tiny_patch4_window7_224")
    eval_metric : str
        Evaluation metric to test ("log_loss" or "f1_micro")
    """
    # Set up data and model
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, _ = shopee_dataset(download_dir)
    model_path = f"./tmp/{uuid.uuid4().hex}-automm_shopee"

    predictor = MultiModalPredictor(label="label", problem_type="multiclass", eval_metric=eval_metric, path=model_path)

    # Train the model
    predictor.fit(
        hyperparameters={
            "model.mmdet_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": -1,
            "optimization.max_epochs": 1,
        },
        train_data=train_data,
        time_limit=30,  # seconds
    )

    # Get predictions
    if eval_metric == "log_loss":
        y_pred = predictor.predict_proba(train_data)
        y_true = train_data[predictor.label].values
        manual_score = log_loss(y_true, y_pred)
    elif eval_metric == "f1_micro":
        y_pred = predictor.predict(train_data)
        y_true = train_data[predictor.label].values
        manual_score = f1_score(y_true, y_pred, average="micro")
    else:
        raise NotImplementedError

    # Get score from predictor's evaluate method
    predictor_score = predictor.evaluate(train_data)

    # Verify metric configuration
    assert predictor.eval_metric == eval_metric

    # Verify scores match (within numerical precision)
    np.testing.assert_almost_equal(
        predictor_score[eval_metric],
        manual_score,
        decimal=5,
        err_msg=f"Predictor's {eval_metric} score doesn't match manual calculation",
    )

    # Verify score is within reasonable bounds
    if eval_metric == "log_loss":
        assert predictor_score[eval_metric] > 0, "Log loss should be positive"
    else:  # f1_micro
        assert 0 <= predictor_score[eval_metric] <= 1, "F1 score should be between 0 and 1"
