import os
import shutil

import onnxruntime as ort
import pytest
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances

from autogluon.multimodal import MultiModalOnnxPredictor, MultiModalPredictor


def evaluate(predictor, df, onnx_session=None):
    labels = df["score"].to_numpy()

    if not onnx_session:
        QEmb = predictor.extract_embedding(df[["sentence1"]])["sentence1"]
        AEmb = predictor.extract_embedding(df[["sentence2"]])["sentence2"]
    else:
        valid_input = [
            "hf_text_text_token_ids",
            "hf_text_text_valid_length",
            "hf_text_text_segment_ids",
        ]
        QEmb = onnx_session.run(
            None, predictor.get_processed_batch_for_deployment(data=df[["sentence1"]], valid_input=valid_input)
        )[0]
        AEmb = onnx_session.run(
            None, predictor.get_processed_batch_for_deployment(data=df[["sentence2"]], valid_input=valid_input)
        )[0]

    cosine_scores = 1 - paired_cosine_distances(QEmb, AEmb)
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    return eval_pearson_cosine, eval_spearman_cosine


@pytest.mark.parametrize(
    "checkpoint_name",
    ["sentence-transformers/msmarco-MiniLM-L-12-v3", "sentence-transformers/all-MiniLM-L6-v2"],
)
@pytest.mark.skip(reason="onnx export currently requires torchtext<=0.12.0")
def test_onnx_export(checkpoint_name):
    test_df = load_dataset("wietsedv/stsbenchmark", split="test").to_pandas()

    predictor = MultiModalPredictor(
        problem_type="feature_extraction",
        hyperparameters={
            "model.hf_text.checkpoint_name": checkpoint_name,
        },
    )
    ag_pearson, ag_spearman = evaluate(predictor, test_df)

    onnx_path = checkpoint_name.replace("/", "_") + ".onnx"

    predictor.export_onnx(onnx_path=onnx_path, data=test_df)
    ort_sess = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    onnx_pearson, onnx_spearman = evaluate(predictor, test_df, ort_sess)
    assert pytest.approx(onnx_pearson, 1e-2) == ag_pearson
    assert pytest.approx(onnx_spearman, 1e-2) == ag_spearman


@pytest.mark.parametrize(
    "checkpoint_name,num_gpus",
    [
        ("swin_tiny_patch4_window7_224", -1),
        ("vit_tiny_patch16_224", -1),
        ("resnet18", 0),
        ("legacy_seresnet18", -1),
    ],
)
def test_onnx_export_image_cls(checkpoint_name, num_gpus):
    import numpy as np
    import torch
    from torch import FloatTensor

    from autogluon.multimodal.utils import logits_to_prob
    from autogluon.multimodal.utils.misc import shopee_dataset

    model_path = "./automm_shopee"
    download_dir = "./ag_automm_tutorial_imgcls"
    train_data, test_data = shopee_dataset(download_dir)
    image_path_export = test_data.iloc[0]["image"]
    image_path_test = test_data.iloc[1]["image"]

    if os.path.exists(model_path):
        shutil.rmtree(model_path)

    # train
    predictor = MultiModalPredictor(
        hyperparameters={
            "model.names": ["timm_image"],
            "model.timm_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
        },
        label="label",
        path=model_path,
    )
    predictor.fit(
        train_data=train_data,
        time_limit=20,  # seconds
    )
    predictor.save(path=model_path)
    loaded_predictor = MultiModalPredictor.load(path=model_path)

    # predict
    load_proba = loaded_predictor.predict_proba({"image": [image_path_test]})

    # convert
    loaded_predictor.export_onnx({"image": [image_path_export]})

    # onnx predict
    onnx_predictor = MultiModalOnnxPredictor.load(model_path)
    onnx_proba = onnx_predictor.predict_proba({"image": [image_path_test]})

    # assert allclose
    np.testing.assert_allclose(load_proba, onnx_proba, rtol=1e-3, atol=1e-3)
