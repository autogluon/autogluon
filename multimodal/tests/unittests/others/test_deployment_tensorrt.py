import os
import shutil
import uuid

import numpy.testing
import pytest
from datasets import load_dataset
from torch.jit._script import RecursiveScriptModule

from autogluon.multimodal import MultiModalPredictor

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

    # Subsample test data for efficient testing, and use a different subset of the dataset for compilation.
    tail_df = dataset.test_df.tail(2)
    test_df_b2 = dataset.test_df.head(2)
    test_df_b4 = dataset.test_df.head(4)
    test_df_b8 = dataset.test_df.head(8)

    # Prediction with default predictor
    y_pred_b2 = predictor.predict(test_df_b2)
    y_pred_b4 = predictor.predict(test_df_b4)
    y_pred_b8 = predictor.predict(test_df_b8)

    predictor.optimize_for_inference(data=tail_df)

    # Prediction with tensorrt predictor
    y_pred_trt_b2 = predictor.predict(test_df_b2)
    y_pred_trt_b4 = predictor.predict(test_df_b4)
    y_pred_trt_b8 = predictor.predict(test_df_b8)

    assert isinstance(
        predictor._model, OnnxModule
    ), f"invalid onnx module type, expected to be OnnxModule, but the model type is {type(predictor._model)}"

    # Verify correctness of results
    numpy.testing.assert_allclose(y_pred_b2, y_pred_trt_b2, rtol=0.002)
    numpy.testing.assert_allclose(y_pred_b4, y_pred_trt_b4, rtol=0.002)
    numpy.testing.assert_allclose(y_pred_b8, y_pred_trt_b8, rtol=0.002)
