import os
import shutil

import numpy as np
import numpy.testing
import pytest
from packaging import version
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import paired_cosine_distances

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import REGRESSION
from autogluon.multimodal.utils.misc import shopee_dataset
from autogluon.multimodal.utils.onnx import OnnxModule

from ..utils import AEDataset, PetFinderDataset

ALL_DATASETS = {
    "petfinder": PetFinderDataset(),
    "ae": AEDataset(),
}

try:
    import tensorrt
except ImportError:
    tensorrt = None


def evaluate(predictor, df, onnx_session=None):
    labels = df["score"].to_numpy()

    if not onnx_session:
        QEmb = predictor.extract_embedding(df[["sentence1"]])["sentence1"]
        AEmb = predictor.extract_embedding(df[["sentence2"]])["sentence2"]
    else:
        QEmb = onnx_session.run(None, predictor._learner.get_processed_batch_for_deployment(data=df[["sentence1"]]))[0]
        AEmb = onnx_session.run(None, predictor._learner.get_processed_batch_for_deployment(data=df[["sentence2"]]))[0]

    cosine_scores = 1 - paired_cosine_distances(QEmb, AEmb)
    eval_pearson_cosine, _ = pearsonr(labels, cosine_scores)
    eval_spearman_cosine, _ = spearmanr(labels, cosine_scores)

    return eval_pearson_cosine, eval_spearman_cosine


@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "checkpoint_name",
    ["sentence-transformers/msmarco-MiniLM-L-12-v3", "sentence-transformers/all-MiniLM-L6-v2"],
)
def test_onnx_export_hf_text(checkpoint_name):
    # IMPORTANT: lazy import onnxruntime, otherwise onnxruntime won't be able to compile to tensorrt EP.
    import onnxruntime as ort
    import pandas as pd

    # https://huggingface.co/datasets/stsb_multi_mt/viewer/en/test
    data_dict = {
        "sentence1": [
            "A girl is styling her hair.",
            "A group of men play soccer on the beach.",
            "One woman is measuring another woman's ankle.",
            "A man is cutting up a cucumber.",
            "A man is playing a harp.",
        ],
        "sentence2": [
            "A girl is brushing her hair.",
            "A group of boys are playing soccer on the beach.",
            "A woman measures another woman's ankle.",
            "A man is slicing a cucumber.",
            "A man is playing a keyboard.",
        ],
        "score": [2.5, 3.6, 5, 4.2, 1.5],
    }
    test_df = pd.DataFrame.from_dict(data_dict)

    predictor = MultiModalPredictor(
        problem_type="feature_extraction",
        hyperparameters={
            "optim.max_epochs": 1,
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


@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "checkpoint_name,num_gpus",
    [
        ("swin_tiny_patch4_window7_224", -1),
        ("resnet18", -1),
    ],
)
def test_onnx_export_timm_image(checkpoint_name, num_gpus):
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
            "optim.max_epochs": 1,
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
        time_limit=10,  # seconds
    )
    predictor.save(path=model_path)
    loaded_predictor = MultiModalPredictor.load(path=model_path)

    # predict
    load_proba = loaded_predictor.predict_proba({"image": [image_path_test]})

    # -------------------------------------------------------
    # convert (no path)
    onnx_path = loaded_predictor.export_onnx({"image": [image_path_export]})
    assert isinstance(onnx_path, bytes), (
        f"export_onnx() method should return onnx bytes directly, if path is not provided."
    )

    # create onnx module for evaluation
    onnx_module = OnnxModule(onnx_path, providers=["CUDAExecutionProvider"])
    onnx_module.input_keys = loaded_predictor._learner._model.input_keys
    onnx_module.prefix = loaded_predictor._learner._model.prefix
    onnx_module.get_output_dict = loaded_predictor._learner._model.get_output_dict

    # simply replace _model in the loaded predictor to predict with onnxruntime
    loaded_predictor._learner._model = onnx_module
    onnx_proba = loaded_predictor.predict_proba({"image": [image_path_test]})

    # assert allclose
    np.testing.assert_allclose(load_proba, onnx_proba, rtol=1e-2, atol=1e-2)

    # -------------------------------------------------------
    # convert (with path)
    loaded_predictor = MultiModalPredictor.load(path=model_path)
    onnx_path = loaded_predictor.export_onnx({"image": [image_path_export]}, path=model_path)

    # Check existence of the exported onnx model file and tensorrt cache files
    onnx_path_expected = os.path.join(model_path, "model.onnx")
    assert isinstance(onnx_path, str), "export_onnx() method should return a string, if path argument is provided."
    assert onnx_path == onnx_path_expected, "onnx_path mismatch"
    assert os.path.exists(onnx_path), f"onnx model file not found at {onnx_path}"

    # create onnx module for evaluation
    onnx_module = OnnxModule(onnx_path, providers=["CUDAExecutionProvider"])
    onnx_module.input_keys = loaded_predictor._learner._model.input_keys
    onnx_module.prefix = loaded_predictor._learner._model.prefix
    onnx_module.get_output_dict = loaded_predictor._learner._model.get_output_dict

    # simply replace _model in the loaded predictor to predict with onnxruntime
    loaded_predictor._learner._model = onnx_module
    onnx_proba = loaded_predictor.predict_proba({"image": [image_path_test]})

    # assert allclose
    np.testing.assert_allclose(load_proba, onnx_proba, rtol=1e-2, atol=1e-2)


@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "dataset_name,model_names,text_backbone,image_backbone",
    [
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "fusion_mlp"],
            "google/electra-small-discriminator",
            "mobilenetv3_small_100",
        ),
        (
            "ae",
            ["numerical_mlp", "hf_text", "fusion_mlp"],
            "google/electra-small-discriminator",
            None,
        ),
    ],
)
@pytest.mark.skipif(
    tensorrt is None or version.parse(tensorrt.__version__) >= version.parse("8.5.4"),
    reason="tensorrt above 8.5.4 cause segfault, but is required to support py311",
)
def test_onnx_optimize_for_inference(dataset_name, model_names, text_backbone, image_backbone):
    dataset = ALL_DATASETS[dataset_name]
    hyperparameters = {
        "optim.max_epochs": 1,
        "model.names": model_names,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
    }
    if text_backbone:
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": text_backbone,
            }
        )
    if image_backbone:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": image_backbone,
            }
        )
    predictor = MultiModalPredictor(
        label=dataset.label_columns[0], problem_type=dataset.problem_type, eval_metric=dataset.metric
    )
    predictor.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=5,
    )
    model_path = predictor.path
    predictor.save(path=model_path)

    # Use a different subset of the dataset for compilation.
    tail_df = dataset.test_df.tail(2)

    # Load a refresh predictor and optimize it for inference
    for providers in [None, ["TensorrtExecutionProvider"], ["CUDAExecutionProvider"], ["CPUExecutionProvider"]]:
        predictor_opt = MultiModalPredictor.load(path=model_path)
        predictor_opt.optimize_for_inference(providers=providers)

        # Check module type of optimized predictor
        assert isinstance(predictor_opt._learner._model, OnnxModule), (
            f"invalid onnx module type, expected to be OnnxModule, but the model type is {type(predictor_opt._learner._model)}"
        )

        # We should support dynamic shape
        for batch_size in [2, 4, 8]:
            test_df = dataset.test_df.head(batch_size)
            if dataset.problem_type == REGRESSION:
                y_pred = predictor.predict(test_df)
                y_pred_trt = predictor_opt.predict(test_df)
            else:
                y_pred = predictor.predict_proba(test_df)
                y_pred_trt = predictor_opt.predict_proba(test_df)
            numpy.testing.assert_allclose(y_pred, y_pred_trt, rtol=0.01, atol=0.01)
