import os
import shutil
import time
import uuid

import numpy.testing
import pytest
from datasets import load_dataset
from torch.jit._script import RecursiveScriptModule

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.constants import REGRESSION

from ..utils.unittest_datasets import AEDataset, PetFinderDataset

ALL_DATASETS = {
    "petfinder": PetFinderDataset(),
    "ae": AEDataset(),
}


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
def test_tensorrt_export_hf_text(dataset_name, model_names, text_backbone, image_backbone):
    import torch

    from autogluon.multimodal.utils.onnx import OnnxModule

    dataset = ALL_DATASETS[dataset_name]
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": model_names,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
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
    predictor_trt = MultiModalPredictor.load(path=model_path)
    predictor_trt.optimize_for_inference(data=tail_df)
    onnx_path = os.path.join(predictor_trt.path, "model.onnx")
    assert os.path.exists(onnx_path), f"onnx model file not found at {onnx_path}"
    trt_cache_dir = os.path.join(predictor_trt.path, "model_trt")
    assert len(os.listdir(trt_cache_dir)) >= 2, f"tensorrt cache model files are not found in {trt_cache_dir}"
    assert isinstance(
        predictor_trt._model, OnnxModule
    ), f"invalid onnx module type, expected to be OnnxModule, but the model type is {type(predictor._model)}"

    # We should support dynamic shape
    for batch_size in [2, 4, 8]:
        test_df = dataset.test_df.head(batch_size)
        if dataset.problem_type == REGRESSION:
            y_pred = predictor.predict(test_df)
            y_pred_trt = predictor_trt.predict(test_df)
        else:
            y_pred = predictor.predict_proba(test_df)
            y_pred_trt = predictor_trt.predict_proba(test_df)
        numpy.testing.assert_allclose(y_pred, y_pred_trt, rtol=0.01, atol=0.01)
