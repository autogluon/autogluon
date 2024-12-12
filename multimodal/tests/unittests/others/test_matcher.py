import os
import shutil
import tempfile

import numpy.testing as npt
import pytest

from autogluon.multimodal import MultiModalPredictor
from autogluon.multimodal.utils import convert_data_for_ranking, semantic_search

from ..utils import (
    Flickr30kDataset,
    IDChangeDetectionDataset,
    evaluate_matcher_ranking,
    get_home_dir,
    verify_matcher_realtime_inference,
    verify_matcher_save_load,
)

ALL_DATASETS = {
    "id_change_detection": IDChangeDetectionDataset,
    "flickr30k": Flickr30kDataset,
}


@pytest.mark.parametrize(
    "dataset_name,query,response,problem_type,text_backbone,image_backbone, is_ranking, symmetric",
    [
        (
            "id_change_detection",
            "Previous Image",
            "Current Image",
            "image_similarity",
            None,
            "swin_tiny_patch4_window7_224",
            False,
            False,
        ),
        (
            "flickr30k",
            "caption",
            "image",
            "image_text_similarity",
            "google/electra-small-discriminator",
            "swin_tiny_patch4_window7_224",
            True,
            True,
        ),
    ],
)
def test_matcher_basic(
    dataset_name,
    query,
    response,
    problem_type,
    text_backbone,
    image_backbone,
    is_ranking,
    symmetric,
):
    dataset = ALL_DATASETS[dataset_name]()

    matcher = MultiModalPredictor(
        query=query,
        response=response,
        problem_type=problem_type,
        label=dataset.label_columns[0] if dataset.label_columns else None,
        match_label=dataset.match_label,
        eval_metric=dataset.metric,
    )

    hyperparameters = {
        "optim.max_epochs": 1,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "optim.top_k_average_method": "greedy_soup",
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
        tuning_data=dataset.val_df if hasattr(dataset, "val_df") else None,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )

    if is_ranking:
        evaluate_matcher_ranking(
            matcher=matcher,
            test_df=dataset.test_df,
            query_column=query,
            response_column=response,
            metric_name=dataset.metric,
            symmetric=symmetric,
        )
        text_to_image_hits = semantic_search(
            matcher=matcher,
            query_data={
                query: dataset.test_df[query].tolist()
            },  # need a dict/dataframe instead of a list for a trained matcher
            response_data={
                response: dataset.test_df[response].tolist()
            },  # need a dict/dataframe instead of a list for a trained matcher
            top_k=5,
        )
        image_to_text_hits = semantic_search(
            matcher=matcher,
            query_data={
                response: dataset.test_df[response].tolist()
            },  # need a dict/dataframe instead of a list for a trained matcher
            response_data={
                query: dataset.test_df[query].tolist()
            },  # need a dict/dataframe instead of a list for a trained matcher
            top_k=5,
        )
    else:
        score = matcher.evaluate(dataset.test_df)
    verify_matcher_save_load(matcher, dataset.test_df, cls=MultiModalPredictor)

    # Test for continuous fit
    matcher.fit(
        train_data=dataset.train_df,
        tuning_data=dataset.val_df if hasattr(dataset, "val_df") else None,
        hyperparameters=hyperparameters,
        time_limit=20,
    )
    verify_matcher_save_load(matcher, dataset.test_df, cls=MultiModalPredictor)

    # Saving to folder, loading the saved model and call fit again (continuous fit)
    with tempfile.TemporaryDirectory() as root:
        matcher.save(root)
        matcher = MultiModalPredictor.load(root)
        matcher.fit(
            train_data=dataset.train_df,
            tuning_data=dataset.val_df if hasattr(dataset, "val_df") else None,
            hyperparameters=hyperparameters,
            time_limit=20,
        )


# Conflicts between realtime inference with dp and lightning strategies
@pytest.mark.single_gpu
@pytest.mark.parametrize(
    "dataset_name,query,response,problem_type,text_backbone,image_backbone, is_ranking, symmetric",
    [
        (
            "id_change_detection",
            "Previous Image",
            "Current Image",
            "image_similarity",
            None,
            "swin_tiny_patch4_window7_224",
            False,
            False,
        ),
        (
            "flickr30k",
            "caption",
            "image",
            "image_text_similarity",
            "google/electra-small-discriminator",
            "swin_tiny_patch4_window7_224",
            True,
            True,
        ),
    ],
)
def test_matcher_realtime_inference(
    dataset_name,
    query,
    response,
    problem_type,
    text_backbone,
    image_backbone,
    is_ranking,
    symmetric,
):
    dataset = ALL_DATASETS[dataset_name]()

    matcher = MultiModalPredictor(
        query=query,
        response=response,
        problem_type=problem_type,
        label=dataset.label_columns[0] if dataset.label_columns else None,
        match_label=dataset.match_label,
        eval_metric=dataset.metric,
    )

    hyperparameters = {
        "optim.max_epochs": 1,
        "env.num_workers": 0,
        "env.num_workers_inference": 0,
        "optim.top_k_average_method": "greedy_soup",
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
        tuning_data=dataset.val_df if hasattr(dataset, "val_df") else None,
        hyperparameters=hyperparameters,
        time_limit=20,
        save_path=save_path,
    )

    verify_matcher_realtime_inference(matcher, dataset.test_df)


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

    matcher = MultiModalPredictor(
        problem_type="text_similarity",
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
            {"response_id": 0, "score": 0.7035943269729614},
            {"response_id": 1, "score": 0.527172327041626},
            {"response_id": 3, "score": 0.18880528211593628},
            {"response_id": 6, "score": 0.10461556911468506},
            {"response_id": 8, "score": 0.09811049699783325},
        ],
        [
            {"response_id": 7, "score": 0.6432060599327087},
            {"response_id": 4, "score": 0.2563084363937378},
            {"response_id": 3, "score": 0.1389961689710617},
            {"response_id": 6, "score": 0.11907944828271866},
            {"response_id": 8, "score": 0.1079183965921402},
        ],
        [
            {"response_id": 8, "score": 0.8252813816070557},
            {"response_id": 0, "score": 0.13986822962760925},
            {"response_id": 7, "score": 0.1292111724615097},
            {"response_id": 6, "score": 0.10977005213499069},
            {"response_id": 3, "score": 0.06506325304508209},
        ],
    ]

    for per_query_hits, per_query_hits_2, per_query_hit_gt in zip(hits, hits_2, hits_gt):
        for per_hit, per_hit_2, per_hit_gt in zip(per_query_hits, per_query_hits_2, per_query_hit_gt):
            assert per_hit["response_id"] == per_hit_2["response_id"] == per_hit_gt["response_id"]
            npt.assert_allclose(per_hit["score"], per_hit_2["score"], 1e-3, 1e-3)
            npt.assert_allclose(per_hit["score"], per_hit_gt["score"], 1e-3, 1e-3)


def test_image_text_semantic_search():
    dataset_name = "flickr30k"
    dataset = ALL_DATASETS[dataset_name]()
    image_list = dataset.test_df["image"].tolist()
    text_list = dataset.test_df["caption"].tolist()

    matcher = MultiModalPredictor(
        problem_type="image_text_similarity",
        hyperparameters={"model.clip.checkpoint_name": "openai/clip-vit-base-patch32"},
    )
    text_to_image_hits = semantic_search(
        matcher=matcher,
        query_data=text_list,
        response_data=image_list,
        top_k=5,
    )

    # extract embeddings first and then do semantic search
    query_embeddings = matcher.extract_embedding(text_list, as_tensor=True)
    response_embeddings = matcher.extract_embedding(image_list, as_tensor=True)
    text_to_image_hits_2 = semantic_search(
        matcher=matcher,
        query_embeddings=query_embeddings,
        response_embeddings=response_embeddings,
        top_k=5,
    )

    for per_query_hits, per_query_hits_2 in zip(text_to_image_hits, text_to_image_hits_2):
        for per_hit, per_hit_2 in zip(per_query_hits, per_query_hits_2):
            assert per_hit["response_id"] == per_hit_2["response_id"]
            npt.assert_almost_equal(per_hit["score"], per_hit_2["score"])

    image_to_text_hits = semantic_search(
        matcher=matcher,
        query_data=image_list,
        response_data=text_list,
        top_k=5,
    )

    # extract embeddings first and then do semantic search
    query_embeddings = matcher.extract_embedding(image_list, as_tensor=True)
    response_embeddings = matcher.extract_embedding(text_list, as_tensor=True)
    image_to_text_hits_2 = semantic_search(
        matcher=matcher,
        query_embeddings=query_embeddings,
        response_embeddings=response_embeddings,
        top_k=5,
    )

    for per_query_hits, per_query_hits_2 in zip(image_to_text_hits, image_to_text_hits_2):
        for per_hit, per_hit_2 in zip(per_query_hits, per_query_hits_2):
            assert per_hit["response_id"] == per_hit_2["response_id"]
            npt.assert_almost_equal(per_hit["score"], per_hit_2["score"])


@pytest.mark.parametrize(
    "hyperparameters",
    [
        {
            "model.names": ["timm_image"],
            "model.timm_image.checkpoint_name": "mobilenetv3_small_100",
        },
    ],
)
def test_matcher_hyperparameters_consistency(hyperparameters):
    dataset = IDChangeDetectionDataset()

    # pass hyperparameters to init()
    predictor = MultiModalPredictor(
        query="Previous Image",
        response="Current Image",
        problem_type="image_similarity",
        label=dataset.label_columns[0] if dataset.label_columns else None,
        match_label=dataset.match_label,
        eval_metric=dataset.metric,
        hyperparameters=hyperparameters,
    )
    predictor.fit(dataset.train_df, time_limit=10)

    # pass hyperparameters to fit()
    predictor_2 = MultiModalPredictor(
        query="Previous Image",
        response="Current Image",
        problem_type="image_similarity",
        label=dataset.label_columns[0] if dataset.label_columns else None,
        match_label=dataset.match_label,
        eval_metric=dataset.metric,
    )
    predictor_2.fit(
        dataset.train_df,
        hyperparameters=hyperparameters,
        time_limit=10,
    )
    assert predictor._learner._config == predictor_2._learner._config
