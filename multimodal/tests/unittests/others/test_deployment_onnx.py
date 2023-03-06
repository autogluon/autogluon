import os
import shutil

import onnxruntime as ort
import pytest
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances

from autogluon.multimodal import MultiModalPredictor


def evaluate(predictor, df, onnx_session=None):
    labels = df["score"].to_numpy()

    if not onnx_session:
        QEmb = predictor.extract_embedding(df[["sentence1"]])["sentence1"]
        AEmb = predictor.extract_embedding(df[["sentence2"]])["sentence2"]
    else:
        QEmb = onnx_session.run(None, predictor.get_processed_batch_for_deployment(data=df[["sentence1"]]))[0]
        AEmb = onnx_session.run(None, predictor.get_processed_batch_for_deployment(data=df[["sentence2"]]))[0]

    cosine_scores = 1 - paired_cosine_distances(QEmb, AEmb)
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    return eval_pearson_cosine, eval_spearman_cosine


@pytest.mark.parametrize(
    "checkpoint_name",
    ["sentence-transformers/msmarco-MiniLM-L-12-v3", "sentence-transformers/all-MiniLM-L6-v2"],
)
def test_onnx_export_hf_text(checkpoint_name):
    test_df = load_dataset("wietsedv/stsbenchmark", split="test").to_pandas()
    test_df = test_df.head()  # subsample the data to avoid OOM in tracing

    predictor = MultiModalPredictor(
        problem_type="feature_extraction",
        hyperparameters={
            "optimization.max_epochs": 1,
            "model.hf_text.checkpoint_name": checkpoint_name,
        },
    )
    ag_pearson, ag_spearman = evaluate(predictor, test_df)

    model_path = checkpoint_name.replace("/", "_")
    onnx_path = predictor.export_onnx(path=model_path, data=test_df)

    # TODO: Test with CUDA EP when we upgrade CUDA version to 11.7 (along with pytorch v1.13.1).
    # onnxruntime-gpu v1.13.1 require CUDA version >=11.6
    ort_sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_pearson, onnx_spearman = evaluate(predictor, test_df, ort_sess)
    assert pytest.approx(onnx_pearson, 1e-2) == ag_pearson
    assert pytest.approx(onnx_spearman, 1e-2) == ag_spearman


@pytest.mark.parametrize(
    "checkpoint_name,num_gpus",
    [
        ("swin_tiny_patch4_window7_224", -1),
        ("resnet18", 0),
    ],
)
def test_onnx_export_timm_image(checkpoint_name, num_gpus):
    import numpy as np
    import onnx
    import torch
    from torch import FloatTensor

    from autogluon.multimodal.utils import logits_to_prob
    from autogluon.multimodal.utils.misc import shopee_dataset
    from autogluon.multimodal.utils.onnx import OnnxModule

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
            "optimization.max_epochs": 1,
            "model.names": ["timm_image"],
            "model.timm_image.checkpoint_name": checkpoint_name,
            "env.num_gpus": num_gpus,
            "env.num_workers": 0,
            "env.strategy": "ddp",
        },
        label="label",
        path=model_path,
    )
    predictor.fit(
        train_data=train_data,
        time_limit=30,  # seconds
    )
    predictor.save(path=model_path)
    loaded_predictor = MultiModalPredictor.load(path=model_path)

    # predict
    load_proba = loaded_predictor.predict_proba({"image": [image_path_test]})

    # convert
    onnx_path = loaded_predictor.export_onnx({"image": [image_path_export]})

    # create onnx module for evaluation
    onnx_module = OnnxModule(onnx_path, providers=["CUDAExecutionProvider"])
    onnx_module.input_keys = loaded_predictor._model.input_keys
    onnx_module.prefix = loaded_predictor._model.prefix
    onnx_module.get_output_dict = loaded_predictor._model.get_output_dict

    # simply replace _model in the loaded predictor to predict with onnxruntime
    loaded_predictor._model = onnx_module
    onnx_proba = loaded_predictor.predict_proba({"image": [image_path_test]})

    # assert allclose
    np.testing.assert_allclose(load_proba, onnx_proba, rtol=1e-3, atol=1e-3)
