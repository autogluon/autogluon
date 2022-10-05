import copy
import os
import pickle
import shutil
import tempfile

import numpy.testing as npt
import pytest
from unittest_datasets import IDChangeDetectionDataset
from utils import get_home_dir

from autogluon.multimodal import MultiModalMatcher
from autogluon.multimodal.constants import BINARY, MULTICLASS, QUERY, RESPONSE, UNIFORM_SOUP

ALL_DATASETS = {
    "id_change_detection": IDChangeDetectionDataset,
}


def verify_matcher_save_load(matcher, df, verify_embedding=True, cls=MultiModalMatcher):
    with tempfile.TemporaryDirectory() as root:
        matcher.save(root)
        predictions = matcher.predict(df, as_pandas=False)
        loaded_matcher = cls.load(root)
        predictions2 = loaded_matcher.predict(df, as_pandas=False)
        predictions2_df = loaded_matcher.predict(df, as_pandas=True)
        npt.assert_equal(predictions, predictions2)
        npt.assert_equal(predictions2, predictions2_df.to_numpy())
        if matcher.problem_type in [BINARY, MULTICLASS]:
            predictions_prob = matcher.predict_proba(df, as_pandas=False)
            predictions2_prob = loaded_matcher.predict_proba(df, as_pandas=False)
            predictions2_prob_df = loaded_matcher.predict_proba(df, as_pandas=True)
            npt.assert_equal(predictions_prob, predictions2_prob)
            npt.assert_equal(predictions2_prob, predictions2_prob_df.to_numpy())
        if verify_embedding:
            query_embeddings = matcher.extract_embedding(df, signature=QUERY)
            response_embeddings = matcher.extract_embedding(df, signature=RESPONSE)
            assert query_embeddings.shape[0] == len(df)
            assert response_embeddings.shape[0] == len(df)


@pytest.mark.parametrize(
    "dataset_name,query,response,presets,text_backbone,image_backbone",
    [
        (
            "id_change_detection",
            "Previous Image",
            "Current Image",
            "siamese_network",
            None,
            "swin_tiny_patch4_window7_224",
        ),
    ],
)
def test_matcher(
    dataset_name,
    query,
    response,
    presets,
    text_backbone,
    image_backbone,
):
    dataset = ALL_DATASETS[dataset_name]()

    matcher = MultiModalMatcher(
        query=query,
        response=response,
        label=dataset.label_columns[0],
        match_label=dataset.match_label,
        problem_type=dataset.problem_type,
        eval_metric=dataset.metric,
    )

    hyperparameters = {
        "optimization.max_epochs": 1,
        "env.num_workers": 0,
        "env.num_workers_evaluation": 0,
    }

    if text_backbone is not None:
        hyperparameters.update(
            {
                "model.hf_text.checkpoint_name": text_backbone,
            }
        )
    if image_backbone is not None:
        hyperparameters.update(
            {
                "model.timm_image.checkpoint_name": image_backbone,
            }
        )

    save_path = os.path.join(get_home_dir(), "outputs", dataset_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)

    matcher.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=30,
        save_path=save_path,
    )

    score = matcher.evaluate(dataset.test_df)
    verify_matcher_save_load(matcher, dataset.test_df, cls=MultiModalMatcher)

    # Test for continuous fit
    matcher.fit(
        train_data=dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=30,
    )
    verify_matcher_save_load(matcher, dataset.test_df, cls=MultiModalMatcher)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        matcher.save(root)
        matcher = MultiModalMatcher.load(root)
        matcher.fit(
            train_data=dataset.train_df,
            hyperparameters=hyperparameters,
            time_limit=30,
        )
