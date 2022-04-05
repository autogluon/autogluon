import os
import json
import pytest
import numpy.testing as npt
import tempfile
import copy

from autogluon.text.automm import AutoMMPredictor
from autogluon.text.automm.constants import (
    MODEL,
    DATA,
    OPTIMIZATION,
    ENVIRONMENT,
    BINARY,
    MULTICLASS,
    UNION_SOUP,
    GREEDY_SOUP,
    BEST_SOUP,
)
from datasets import (
    PetFinderDataset,
    HatefulMeMesDataset,
    AEDataset,
)
from utils import get_home_dir

ALL_DATASETS = {
    "petfinder": PetFinderDataset,
    "hateful_memes": HatefulMeMesDataset,
    "ae": AEDataset,
}


def verify_predictor_save_load(predictor, df,
                               verify_embedding=True):
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictions = predictor.predict(df, as_pandas=False)
        loaded_predictor = AutoMMPredictor.load(root)
        predictions2 = loaded_predictor.predict(df, as_pandas=False)
        predictions2_df = loaded_predictor.predict(df, as_pandas=True)
        npt.assert_equal(predictions, predictions2)
        npt.assert_equal(predictions2,
                         predictions2_df.to_numpy())
        if predictor.problem_type in [BINARY, MULTICLASS]:
            predictions_prob = predictor.predict_proba(df, as_pandas=False)
            predictions2_prob = loaded_predictor.predict_proba(df, as_pandas=False)
            predictions2_prob_df = loaded_predictor.predict_proba(df, as_pandas=True)
            npt.assert_equal(predictions_prob, predictions2_prob)
            npt.assert_equal(predictions2_prob, predictions2_prob_df.to_numpy())
        if verify_embedding:
            embeddings = predictor.extract_embedding(df)
            assert embeddings.shape[0] == len(df)


@pytest.mark.parametrize(
    "dataset_name,model_names,text_backbone,image_backbone,top_k_average_method",
    [
        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
            "prajjwal1/bert-tiny",
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP
        ),

        (
            "hateful_memes",
            ["timm_image", "hf_text", "clip", "fusion_mlp"],
            "monsoon-nlp/hindi-bert",
            "swin_tiny_patch4_window7_224",
            UNION_SOUP
        ),

        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "timm_image", "fusion_mlp"],
            None,
            "swin_tiny_patch4_window7_224",
            GREEDY_SOUP
        ),

        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "hf_text", "fusion_mlp"],
            "prajjwal1/bert-tiny",
            None,
            UNION_SOUP
        ),

        (
            "petfinder",
            ["numerical_mlp", "categorical_mlp", "fusion_mlp"],
            None,
            None,
            BEST_SOUP
        ),

        (
            "hateful_memes",
            ["timm_image"],
            None,
            "swin_tiny_patch4_window7_224",
            UNION_SOUP
        ),

        (
            "ae",
            ["hf_text"],
            "prajjwal1/bert-tiny",
            None,
            BEST_SOUP
        ),

        (
            "hateful_memes",
            ["clip"],
            None,
            None,
            BEST_SOUP
        ),

    ]
)
def test_predictor(
        dataset_name,
        model_names,
        text_backbone,
        image_backbone,
        top_k_average_method,
):
    dataset = ALL_DATASETS[dataset_name]()
    metric_name = dataset.metric

    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": model_names,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
        "optimization.top_k_average_method": top_k_average_method,
    }
    if text_backbone is not None:
        hyperparameters.update({
            "model.hf_text.checkpoint_name": text_backbone,
        })
    if image_backbone is not None:
        hyperparameters.update({
            "model.timm_image.checkpoint_name": image_backbone,
        })
    save_path = os.path.join(get_home_dir(), "outputs", dataset_name)
    if text_backbone is not None:
        save_path = os.path.join(save_path, text_backbone)
    if image_backbone is not None:
        save_path = os.path.join(save_path, image_backbone)

    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
    )

    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Test for continuous fit
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=30,
    )
    verify_predictor_save_load(predictor, dataset.test_df)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = AutoMMPredictor.load(root)
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            hyperparameters=hyperparameters,
            time_limit=30,
        )


def test_standalone(): # test standalong feature in AutoMMPredictor.save()
    from unittest import mock
    import torch

    requests_gag = mock.patch(
        'requests.Session.request',
        mock.Mock(side_effect=RuntimeError(
            'Please use the `responses` library to mock HTTP in your tests.'
        ))
    )

    dataset = PetFinderDataset()

    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }

    hyperparameters = {
        "optimization.max_epochs": 1,
        "model.names": ["numerical_mlp", "categorical_mlp", "timm_image", "hf_text", "clip", "fusion_mlp"],
        "model.hf_text.checkpoint_name": "prajjwal1/bert-tiny",
        "model.timm_image.checkpoint_name": "swin_tiny_patch4_window7_224",
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    save_path = os.path.join(get_home_dir(), "standalone", "false")


    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
    )

    save_path_standalone = os.path.join(get_home_dir(), "standalone", "true")

    predictor.save(
        path = save_path_standalone,
        standalone = True
    )

    del predictor
    torch.cuda.empty_cache()

    loaded_online_predictor = AutoMMPredictor.load(path = save_path)
    online_predictions = loaded_online_predictor.predict(dataset.test_df, as_pandas=False)
    del loaded_online_predictor

    # Check if the predictor can be loaded from an offline enivronment.
    with requests_gag:
        # No internet connection here. If any command require internet connection, a RuntimeError will be raised.
        with tempfile.TemporaryDirectory() as tmpdirname:
            torch.hub.set_dir(tmpdirname) # block reading files in `.cache`
            loaded_offline_predictor = AutoMMPredictor.load(path = save_path_standalone)


    offline_predictions = loaded_offline_predictor.predict(dataset.test_df, as_pandas=False)
    del loaded_offline_predictor

    # check if save with standalone=True coincide with standalone=False
    npt.assert_equal(online_predictions,offline_predictions)


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            "model.names": ["timm_image_0", "timm_image_1", "fusion_mlp"],
            "model.timm_image_0.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.timm_image_1.checkpoint_name": "swin_small_patch4_window7_224",
        },

        {
            "model.names": ["hf_text_abc", "hf_text_def", "hf_text_xyz", "fusion_mlp_123"],
            "model.hf_text_def.checkpoint_name": "monsoon-nlp/hindi-bert",
            "model.hf_text_xyz.checkpoint_name": "prajjwal1/bert-tiny",
            "model.hf_text_abc.checkpoint_name": "roberta-base",
        },

        {
            "model.names": ["timm_image_haha", "hf_text_hello", "numerical_mlp_456", "categorical_mlp_abc", "fusion_mlp"],
            "model.timm_image_haha.checkpoint_name": "swin_tiny_patch4_window7_224",
            "model.hf_text_hello.checkpoint_name": "prajjwal1/bert-tiny",
            "data.categorical.convert_to_text": False,
        },
    ]
)
def test_customizing_model_names(
        hyperparameters,
):
    dataset = ALL_DATASETS["petfinder"]()
    metric_name = dataset.metric

    predictor = AutoMMPredictor(
        label=dataset.label_columns[0],
        problem_type=dataset.problem_type,
        eval_metric=metric_name,
    )
    config = {
        MODEL: f"fusion_mlp_image_text_tabular",
        DATA: "default",
        OPTIMIZATION: "adamw",
        ENVIRONMENT: "default",
    }
    hyperparameters.update(
        {
            "env.num_workers": 0,
            "env.num_workers_evaluation": 0,
        }
    )
    save_path = os.path.join(get_home_dir(), "outputs", "petfinder")
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=10,
        save_path=save_path,
    )
    assert sorted(predictor._config.model.names) == sorted(hyperparameters["model.names"])
    for per_name in hyperparameters["model.names"]:
        assert hasattr(predictor._config.model, per_name)

    score = predictor.evaluate(dataset.test_df)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Test for continuous fit
    predictor.fit(
        train_data=dataset.train_df,
        config=config,
        hyperparameters=hyperparameters,
        time_limit=10,
    )
    assert sorted(predictor._config.model.names) == sorted(hyperparameters["model.names"])
    for per_name in hyperparameters["model.names"]:
        assert hasattr(predictor._config.model, per_name)
    verify_predictor_save_load(predictor, dataset.test_df)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        predictor.save(root)
        predictor = AutoMMPredictor.load(root)
        predictor.fit(
            train_data=dataset.train_df,
            config=config,
            hyperparameters=hyperparameters,
            time_limit=10,
        )
        assert sorted(predictor._config.model.names) == sorted(hyperparameters["model.names"])
        for per_name in hyperparameters["model.names"]:
            assert hasattr(predictor._config.model, per_name)
