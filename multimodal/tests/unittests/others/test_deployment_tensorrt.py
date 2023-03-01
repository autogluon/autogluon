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

    batch_size = 2

    # Subsample test data for efficient testing, and use a different subset of the dataset for compilation.
    test_df = dataset.test_df.head(batch_size)
    tail_df = dataset.test_df.tail(batch_size)

    # Prediction with default predictor
    y_pred = predictor.predict(test_df)

    trt_module = predictor.export_tensorrt(path=model_path, data=tail_df, batch_size=batch_size)

    # To use the TensorRT module for prediction, simply replace the _model in the predictor
    predictor._model = trt_module

    # Prediction with tensorrt predictor
    y_pred_trt = predictor.predict(test_df)

    # TODO: caching tensorrt engine
    assert isinstance(
        predictor._model, OnnxModule
    ), f"invalid tensorrt module type, expected to be RecursiveScriptModule, but the model type is {type(predictor._model)}"
    numpy.testing.assert_allclose(y_pred, y_pred_trt, rtol=0.002, atol=0.1)
