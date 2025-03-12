import os
import shutil
import tempfile
from unittest import mock

import numpy.testing as npt
import pandas as pd
import pytest
import torch
from omegaconf import OmegaConf

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import BINARY, FEW_SHOT_CLASSIFICATION, MULTICLASS
from autogluon.multimodal.utils.misc import shopee_dataset

from ..utils import (
    get_home_dir,
    verify_predict_and_predict_proba,
    verify_predict_as_pandas_and_multiclass,
    verify_predict_without_label_column,
    verify_predictor_realtime_inference,
    verify_predictor_save_load,
)


@pytest.mark.single_gpu
def test_few_shot_svm_fit_predict():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    save_path = f"./tmp/automm_stanfordcars-8shot-en"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor = MultiModalPredictor(
        label="label",
        problem_type=FEW_SHOT_CLASSIFICATION,
        hyperparameters={
            "model.clip.checkpoint_name": "openai/clip-vit-base-patch32",
            "model.clip.image_size": 224,
        },
        path=save_path,
    )
    predictor.fit(train_data)
    verify_predictor_save_load(predictor, test_data, verify_embedding=True)
    verify_predictor_realtime_inference(predictor, test_data, verify_embedding=True)
    verify_predict_without_label_column(test_data, predictor)
    verify_predict_and_predict_proba(test_data, predictor)
    verify_predict_as_pandas_and_multiclass(test_data, predictor)


def test_few_shot_svm_save_load():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    save_path = f"./tmp/automm_stanfordcars-8shot-en"
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor = MultiModalPredictor(
        label="label",
        problem_type=FEW_SHOT_CLASSIFICATION,
        hyperparameters={
            "model.clip.checkpoint_name": "openai/clip-vit-base-patch32",
            "model.clip.image_size": 224,
        },
        path=save_path,
    )
    predictor.fit(train_data)
    results = predictor.evaluate(test_data)
    preds = predictor.predict(test_data.drop(columns=["label"], axis=1))
    predictor2 = MultiModalPredictor.load(save_path)
    results2 = predictor2.evaluate(test_data)
    preds2 = predictor2.predict(test_data.drop(columns=["label"], axis=1))
    assert results == results2
    assert (preds == preds2).all()
    predictor2.fit(train_data)


@pytest.mark.parametrize(
    "hyperparameters,gt_ckpt_name,gt_model_name",
    [
        (
            {
                "model.names": ["hf_text", "timm_image"],
                "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
            },
            "swin_small_patch4_window7_224",
            ["timm_image"],
        ),
        (
            {
                "model.names": ["timm_image"],
                "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
            },
            "swin_small_patch4_window7_224",
            ["timm_image"],
        ),
        (
            {
                "model.names": ["clip", "timm_image"],
                "model.timm_image.checkpoint_name": "swin_small_patch4_window7_224",
            },
            "swin_small_patch4_window7_224",
            ["timm_image"],
        ),
        (
            {
                "model.names": ["clip", "hf_text"],
                "model.clip.checkpoint_name": "openai/clip-vit-base-patch32",
            },
            "openai/clip-vit-base-patch32",
            ["clip"],
        ),
    ],
)
def test_few_shot_customize_models(hyperparameters, gt_ckpt_name, gt_model_name):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    predictor = MultiModalPredictor(
        label="label",
        problem_type=FEW_SHOT_CLASSIFICATION,
        hyperparameters=hyperparameters,
    )
    predictor.fit(train_data)
    assert predictor._learner._config.model.names == gt_model_name
    assert OmegaConf.select(predictor._learner._config.model, f"{gt_model_name[0]}.checkpoint_name") == gt_ckpt_name


def test_one_shot_two_classes():
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    predictor = MultiModalPredictor(
        label="label",
        problem_type=FEW_SHOT_CLASSIFICATION,
        hyperparameters={
            "model.clip.checkpoint_name": "openai/clip-vit-base-patch32",
            "model.clip.image_size": 224,
        },
    )
    train_data = train_data.groupby("label").sample(n=1)[:2]
    test_data = test_data.groupby("label").sample(n=1)[:2]
    assert len(train_data) == 2 and len(test_data) == 2
    assert len(train_data["label"].unique()) == 2 and len(test_data["label"].unique()) == 2
    predictor.fit(train_data)
    score = predictor.evaluate(test_data)
    pred = predictor.predict(test_data.drop(columns=["label"], axis=1))
    prob = predictor.predict_proba(test_data.drop(columns=["label"], axis=1))
    embedding = predictor.extract_embedding(test_data.drop(columns=["label"], axis=1))


@pytest.mark.parametrize(
    "column_features_pooling_mode",
    ["concat", "mean"],
)
def test_few_shot_multi_columns(column_features_pooling_mode):
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    train_data = pd.concat([train_data["image"]] * 3 + [train_data["label"]], axis=1, ignore_index=True)
    train_data.rename(
        dict(zip(train_data.columns, ["image_1", "image_2", "image_3", "label"])),
        axis=1,
        inplace=True,
    )
    test_data = pd.concat([test_data["image"]] * 3 + [test_data["label"]], axis=1, ignore_index=True)
    test_data.rename(
        dict(zip(test_data.columns, ["image_1", "image_2", "image_3", "label"])),
        axis=1,
        inplace=True,
    )
    assert len(train_data.columns) == 4 and len(test_data.columns) == 4
    predictor = MultiModalPredictor(
        label="label",
        problem_type=FEW_SHOT_CLASSIFICATION,
        hyperparameters={
            "model.clip.checkpoint_name": "openai/clip-vit-base-patch32",
            "model.clip.image_size": 224,
            "data.column_features_pooling_mode": column_features_pooling_mode,
        },
    )
    predictor.fit(train_data)
    score = predictor.evaluate(test_data)
    pred = predictor.predict(test_data.drop(columns=["label"], axis=1))
    proba = predictor.predict_proba(test_data.drop(columns=["label"], axis=1))
    embedding = predictor.extract_embedding(test_data.drop(columns=["label"], axis=1))


def test_few_shot_standalone():  # test standalone feature in MultiModalPredictor.save()
    requests_gag = mock.patch(
        "requests.Session.request",
        mock.Mock(side_effect=RuntimeError("Please use the `responses` library to mock HTTP in your tests.")),
    )
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    predictor = MultiModalPredictor(
        label="label",
        problem_type=FEW_SHOT_CLASSIFICATION,
        hyperparameters={
            "model.clip.checkpoint_name": "openai/clip-vit-base-patch32",
            "model.clip.image_size": 224,
        },
    )
    save_path = os.path.join(get_home_dir(), "outputs", "few_shot_standalone", "true")
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    predictor.fit(
        train_data=train_data,
        save_path=save_path,
        standalone=True,
    )
    predictions = predictor.predict(test_data, as_pandas=False)
    torch.cuda.empty_cache()
    # Check if the predictor can be loaded from an offline environment.
    with requests_gag:
        # No internet connection here. If any command require internet connection, a RuntimeError will be raised.
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.hub.set_dir(tmpdirname)  # block reading files in `.cache`
            loaded_offline_predictor = MultiModalPredictor.load(path=save_path)

    offline_predictions = loaded_offline_predictor.predict(test_data, as_pandas=False)
    npt.assert_equal(predictions, offline_predictions)
