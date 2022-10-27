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
from autogluon.multimodal.utils import semantic_search

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


def test_text_semantic_search():
    corpus = [
        "A man is eating food.",
        "A man is eating a piece of bread.",
        "The girl is carrying a baby.",
        "A man is riding a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "A man is riding a white horse on an enclosed ground.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]
    queries = [
        "A man is eating pasta.",
        "Someone in a gorilla costume is playing a set of drums.",
        "A cheetah chases prey on across a field.",
    ]

    matcher = MultiModalMatcher(
        pipeline="text_similarity",
        hyperparameters={"model.hf_text.checkpoint_name": "sentence-transformers/all-MiniLM-L6-v2"},
    )
    hits = semantic_search(
        matcher=matcher,
        query_data=queries,
        response_data=corpus,
        top_k=5,
    )
    # extract embeddings first and then do semantic search
    query_embeddings = matcher.extract_embedding(queries)
    response_embeddings = matcher.extract_embedding(corpus)
    hits_2 = semantic_search(
        matcher=matcher,
        query_embeddings=query_embeddings,
        response_embeddings=response_embeddings,
        top_k=5,
    )

    hits_gt = [
        [
            {"corpus_id": 0, "score": 0.7035943269729614},
            {"corpus_id": 1, "score": 0.527172327041626},
            {"corpus_id": 3, "score": 0.18880528211593628},
            {"corpus_id": 6, "score": 0.10461556911468506},
            {"corpus_id": 8, "score": 0.09811049699783325},
        ],
        [
            {"corpus_id": 7, "score": 0.6432060599327087},
            {"corpus_id": 4, "score": 0.2563084363937378},
            {"corpus_id": 3, "score": 0.1389961689710617},
            {"corpus_id": 6, "score": 0.11907944828271866},
            {"corpus_id": 8, "score": 0.1079183965921402},
        ],
        [
            {"corpus_id": 8, "score": 0.8252813816070557},
            {"corpus_id": 0, "score": 0.13986822962760925},
            {"corpus_id": 7, "score": 0.1292111724615097},
            {"corpus_id": 6, "score": 0.10977005213499069},
            {"corpus_id": 3, "score": 0.06506325304508209},
        ],
    ]

    for per_query_hits, per_query_hits_2, per_query_hit_gt in zip(hits, hits_2, hits_gt):
        for per_hit, per_hit_2, per_hit_gt in zip(per_query_hits, per_query_hits_2, per_query_hit_gt):
            assert per_hit["corpus_id"] == per_hit_2["corpus_id"] == per_hit_gt["corpus_id"]
            npt.assert_almost_equal(per_hit["score"], per_hit_2["score"])
            npt.assert_almost_equal(per_hit["score"], per_hit_gt["score"])
